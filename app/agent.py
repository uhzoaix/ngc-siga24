import os, pickle
import os.path as op
import numpy as np
import trimesh
import torch
from time import time
from tqdm.autonotebook import tqdm

import app_utils as utils

class Agent():
    """docstring for Agent."""
    def __init__(self):
        pass

    def __call__(self, name, arg):
        method_name = f'action_{name}'
        if hasattr(self, method_name):
            method = getattr(self, method_name)
        else:
            raise NotImplementedError('Not found')
        
        res = method(arg)
        print(f'Done: {method_name}')
        return res
    
    def encode_key(self, shape_name, curve_name):
        return f'{shape_name}|{curve_name}'
    
    def decode_key(self, key):
        return key.split('|')
    
    def curve_from_key(self, key):
        shape_name, curve_name = self.decode_key(key)
        handle = self.handles[shape_name]
        return handle.curve_dict[curve_name]
    
    def load_model(self, device, log_path, checkpoint='final'):
        cpu_device = torch.device('cpu')
        model, opt = utils.load_model(cpu_device, log_path, checkpoint)
        self.model = model
        self.model.to(device)
        self.opt = opt
        self.device = device
    
    def load_data(self, data_root):
        shapes = np.loadtxt(op.join(data_root, 'data.txt'), dtype=str).tolist()
        handles = {}
        feat_dict = {}
        fid = 0
        for shape_name in shapes:
            item_path = op.join(data_root, f'{shape_name}')
            handle_path = op.join(item_path, 'handle/std_handle.pkl')
            handle = utils.load_handle(handle_path)
            handles[shape_name] = handle
            for curve in handle.curves:
                key = f'{shape_name}|{curve.name}'
                key = self.encode_key(shape_name, curve.name)
                feat_dict[key] = fid
                fid += 1

        self.handles = handles
        self.feat_dict = feat_dict

    def load_shape_handle(self, data_root, shape_name):
        item_path = op.join(data_root, f'{shape_name}')
        handle_path = op.join(item_path, 'handle/std_handle.pkl')
        handle = utils.load_handle(handle_path)
        return handle

    def set_embedding(self, device, log_path):
        embd_model, _ = utils.load_model(device, log_path, 'final')
        self.model.set_embedding(embd_model.encoder.embd)

    def apply_transform(self, arg):
        handle = arg['handle']
        # curve posing, by setting new pose of current skeleton
        if 'pose' in arg: 
            pose_file = arg['pose']['pose_file']
            z_axis = None
            if 'z_axis' in arg['pose']:
                z_axis = arg['pose']['z_axis']
                z_axis = np.asarray(z_axis)

            handle.apply_pose(pose_file, z_axis)
        
        # local scaling by changing the key radius
        if 'scaling' in arg:
            handle.apply_scaling(arg['scaling'])

        # tilting, or twisting the shape, by changing the key frame(axis)
        if 'tilt' in arg:
            handle.apply_tilt(arg['tilt'])


    def __inference_vals(self, curve_data, key, batch_size=None):
        # use_batch: aim to divide data into batches to save GPU mem
        num_samples = curve_data['samples'].shape[0]
        if batch_size is not None and num_samples > batch_size:
            N = num_samples // batch_size + 1
            vals = []
            batches = np.array_split(np.arange(num_samples), N)
            for batch in batches:
                batch_curve_data = {key: val[batch] for key,val in curve_data.items()}
                batch_curve_data['device'] = self.device
                batch_curve_data['curve_idx'] = self.feat_dict[key]
                vals_batch = self.model.inference(batch_curve_data).squeeze()
                vals.append(vals_batch.detach().cpu().numpy())
            
            return np.concatenate(vals)

        curve_data['device'] = self.device
        curve_data['curve_idx'] = self.feat_dict[key]

        with torch.no_grad():
            vals = self.model.inference(curve_data).squeeze()
            vals = vals.detach().cpu().numpy()

        return vals
    
    def __mix_inference(self, curve_data, mix_arg, batch_size=None):
        num_samples = curve_data['samples'].shape[0]
        cd = curve_data
        if batch_size is not None and num_samples > batch_size:
            N = num_samples // batch_size + 1
            vals = []
            batches = np.array_split(np.arange(num_samples), N)
            for batch in batches:
                mix_arg['samples_local'] = cd['samples_local'][batch]
                mix_arg['coords'] = cd['coords'][batch]
                vals_batch = self.model.mix_curve(mix_arg).squeeze()
                vals.append(vals_batch.detach().cpu().numpy())

            return np.concatenate(vals)
        else:
            mix_arg['samples_local'] = cd['samples_local']
            mix_arg['coords'] = cd['coords']
            vals = self.model.mix_curve(mix_arg).squeeze()
            return vals.detach().cpu().numpy()


    def shape_repose(self, arg):
        shape_arg = arg['shape']
        shape_name = shape_arg['name']
        handle = self.handles[shape_name]
        if 'pose_file' in shape_arg:
            pose_file = shape_arg['pose_file']
            handle.apply_pose(pose_file)

        if 'rotation' in shape_arg:
            rot_arg = shape_arg['rotation']
            handle.action_rotate_euler(rot_arg)

    def output_mesh(self, mesh, out_name, arg):
        output_folder = op.join(arg['output_path'], arg['config_name'])
        os.makedirs(output_folder, exist_ok=True)

        mesh.export(op.join(output_folder, out_name))
        print('{}|{} Done.'.format(
            arg['exp_name'], arg['config_name']
        ))


    @torch.no_grad()
    def action_ngcnet_inference(self, arg):
        data_root = arg['data_root']
        self.load_data(data_root)
        mc_grid = arg['mc_grid']
        output_folder = arg['output_folder']

        num_shapes = len(self.handles)
        err_res = {}
        reso = mc_grid.reso
        shapes = os.listdir(data_root)
        # max number of query points for Marching Cubes
        batch_size = 32**3
        with tqdm(total=num_shapes) as pbar:
            for shape_name,handle in self.handles.items():
                if shape_name not in shapes:
                    pbar.update(1)
                    continue

                temp_grid = utils.create_grid_like(mc_grid)
                for curve in handle.curves:
                    key = self.encode_key(shape_name, curve.name)
                    curve_data, kidx = curve.filter_grid(mc_grid)
                    
                    vals = self.__inference_vals(curve_data, key, batch_size=batch_size)
                    temp_grid.update_grid(vals, kidx, mode='minimum')
                
                mesh = temp_grid.extract_mesh()
                mesh_file = op.join(output_folder, shape_name, f'mesh{reso}.ply')
                os.makedirs(op.dirname(mesh_file), exist_ok=True)
                mesh.export(mesh_file)
                temp_grid = None

                # gt_file = op.join(data_root, shape_name, 'mesh.ply')
                # err = utils.eval_shape(mesh_file, gt_file)
                # err_res[shape_name] = err

                pbar.update(1)
        
        # err_file = op.join(output_folder, 'err.pkl')
        # with open(err_file, 'wb') as f:
        #     pickle.dump(err_res, f)

    @torch.no_grad()        
    def action_deepsdf_inference(self, arg):
        data_root = arg['data_root']
        mc_grid = arg['mc_grid']
        output_folder = arg['output_folder']

        shapes = np.loadtxt(op.join(data_root, 'data.txt'), dtype=str)
        num_shapes = len(shapes)
        batch_size = 32**3
        num_samples = mc_grid.val_grid.shape[0]
        N = num_samples // batch_size + 1
        kidx_batch = np.array_split(np.arange(num_samples), N)
        with tqdm(total=num_shapes) as pbar:
            for idx in range(num_shapes):
                temp_grid = utils.create_grid_like(mc_grid)
                shape_name = shapes[idx]
                for kidx in kidx_batch:
                    samples = mc_grid.idx2pts(kidx)
                    data = {
                        'samples': torch.from_numpy(samples).float().to(self.device).unsqueeze(0),
                        'idx': torch.LongTensor([idx]).to(self.device)
                    }
                    vals = self.model.inference(data)
                    vals = vals.detach().cpu().numpy()
                    temp_grid.update_grid(vals, kidx)
                
                mesh = temp_grid.extract_mesh()
                mesh_file = op.join(output_folder, shape_name, f'mesh.ply')
                os.makedirs(op.dirname(mesh_file), exist_ok=True)
                mesh.export(mesh_file)
                temp_grid = None

                pbar.update(1)


    @torch.no_grad()
    def action_shape_transform(self, arg):
        data_root = arg['data_root']
        output_folder = arg['output_folder']
        os.makedirs(output_folder, exist_ok=True)
        exp_name = arg['exp_name']
        mc_grid = arg['mc_grid']
        shape_name = arg['shape']
        
        handle = self.load_shape_handle(data_root, shape_name)
        config = utils.load_yaml_file(arg['transform_file'])
        config['handle'] = handle
        self.apply_transform(config)
        out_name = f'{exp_name}_{shape_name}'
        # cyl_outfolder = op.join(output_folder, 'cylinder')
        # os.makedirs(cyl_outfolder, exist_ok=True)
        # handle.export_skeleton_mesh(cyl_outfolder, reso=256)

        batch_size = 32**3
        for curve in handle.curves:
            key = self.encode_key(shape_name, curve.name)
            curve_data, kidx = curve.filter_grid(mc_grid)
            
            vals = self.__inference_vals(curve_data, key, batch_size)
            mc_grid.update_grid(vals, kidx, mode='minimum')

        mesh = mc_grid.extract_mesh()
        mesh_file = op.join(output_folder, f'{out_name}.ply')
        mesh.export(mesh_file)


    @torch.no_grad()
    def action_part_mixing(self, arg):
        output_folder = arg['output_folder']
        exp_name = arg['exp_name']
        mc_grid = arg['mc_grid']
        shape_name = arg['shape']
        config = utils.load_yaml_file(arg['mixing_file'])

        data_root = arg['data_root']
        handle = self.load_shape_handle(data_root, shape_name)
        # out_name = exp_name
        out_name = f'{exp_name}_{shape_name}'

        batch_size = 64**3
        for curve in handle.curves:
            key = self.encode_key(shape_name, curve.name)

            if key in config:
                mix_config = config[key]
                new_key = mix_config['new_key']

                if new_key == 'None':
                    continue
                
                func1 = utils.define_mix_func(mix_config, weights_reverse=True)
                func2 = utils.define_mix_func(mix_config, weights_reverse=False)

                mix_arg = {
                    'curve_handle': self.curve_from_key(new_key),
                    'mix_func1': func1,
                    'mix_func2': func2,
                    'device': self.device,
                    'curve_idx': self.feat_dict[key],
                    'new_idx': self.feat_dict[new_key],
                }

                curve_data, kidx = curve.filter_grid_mix(mc_grid, mix_arg)
                vals = self.__mix_inference(curve_data, mix_arg, batch_size)
            else:
                curve_data, kidx = curve.filter_grid(mc_grid)
                vals = self.__inference_vals(curve_data, key, batch_size)

            mc_grid.update_grid(vals, kidx, mode='minimum')

        mesh = mc_grid.extract_mesh()
        mesh_file = op.join(output_folder, f'{out_name}.ply')
        os.makedirs(op.dirname(mesh_file), exist_ok=True)
        mesh.export(mesh_file)

    @torch.no_grad()
    def action_visualize_SDF(self, arg):
        output_folder = arg['output_folder']
        exp_name = arg['exp_name']
        shape_name = arg['shape']
        handle = self.handles[shape_name]
        
        samples = arg['samples']
        N = samples.shape[0]
        sdfs = 10*np.ones(N)
        mask = np.zeros(N, dtype=bool)

        batch_size = 64**3
        for curve in handle.curves:
            key = self.encode_key(shape_name, curve.name)
            curve_data, inside = curve.localize_samples(samples)
            mask[inside] = True
            
            vals = self.__inference_vals(curve_data, key, batch_size)
            sdfs[inside] = np.minimum(sdfs[inside], vals)

        out_file = op.join(output_folder, 'vis_sdf', f'VisSDF_{shape_name}.png')
        os.makedirs(op.dirname(out_file), exist_ok=True)
        img_size = int(np.sqrt(N))
        utils.sdf2image(out_file, img_size, sdfs, mask, a_max=0.2)
    
    @torch.no_grad()
    def action_shape_manipulate(self, arg):
        output_folder = arg['output_folder']
        exp_name = arg['exp_name']
        mc_grid = arg['mc_grid']
        shape_name = arg['shape']

        handle = self.handles[shape_name]
        # manipuate armadillo
        cR_leg = handle.curve_dict['R_leg']
        cR_foot = handle.curve_dict['R_foot']
        cL_arm = handle.curve_dict['L_arm']
        cL_hand = handle.curve_dict['L_hand']
        idx = 2
        anchor,rot = cR_leg.rot_part(idx, 'z', -45)
        cR_leg.update()
        cR_foot.apply_rotation(anchor, rot)
        cR_foot.update()

        anchor,rot = cL_arm.rot_part(idx, 'y', -30)
        cL_arm.update()
        cL_hand.apply_rotation(anchor, rot)
        cL_hand.update()

        out_name = f'{exp_name}_{shape_name}'
        cyl_folder = op.join(output_folder, out_name)
        os.makedirs(cyl_folder, exist_ok=True)
        handle.export_skeleton_mesh(cyl_folder, reso=256)
        # raise ValueError

        batch_size = 64**3
        for curve in handle.curves:
            key = self.encode_key(shape_name, curve.name)
            curve_data, kidx = curve.filter_grid(mc_grid)
            
            vals = self.__inference_vals(curve_data, key, batch_size)
            mc_grid.update_grid(vals, kidx, mode='minimum')

        t0 = time()
        mesh = mc_grid.extract_mesh()
        print('MC time cost: ', time()-t0)
        mesh_file = op.join(output_folder, f'{exp_name}_{shape_name}.ply')
        os.makedirs(op.dirname(mesh_file), exist_ok=True)
        mesh.export(mesh_file)


    def action_add_part(self, arg):
        mc_grid = arg['mc_grid']
        delta = arg['delta']
        shape_arg = arg['shape']
        new_part_arg = arg['new_part']

        shape_name = shape_arg['name']
        new_shape_name = new_part_arg['shape_name']
        new_part_name = new_part_arg['part_name']
        
        handle = self.handles[shape_name]

        new_handle = self.handles[new_shape_name]
        if 'pose_file' in new_part_arg:
            pose_file = new_part_arg['pose_file']
            new_handle.apply_pose(pose_file)

        if 'rotation' in new_part_arg:
            rot_arg = new_part_arg['rotation']
            vec = rot_arg['vec']
            anchor_idx = rot_arg['anchor_idx']
            new_handle.action_rotate(
                new_part_name, vec, anchor_idx
            )

        smooth = utils.SmoothMaxMin(3, delta)
        new_curve = new_handle.curve_dict[new_part_name]
        new_grid = utils.create_grid_like(mc_grid)

        if arg['area_mode'] == 'large':
            ## Step1: calculate cylinders and blend new part cylinder
            # NOTE: only handle considered, not content(shape).
            for curve in handle.curves:
                # points inside delta-level set of cylinders
                sdfs, kidx = curve.calc_global_implicit(mc_grid, delta)
                # NOTE: use np.minimum for simple boolean union
                mc_grid.update_grid(sdfs, kidx, mode='minimum', mark=True)

            cyl_sdfs, cyl_kidx = new_curve.calc_global_implicit(mc_grid, delta)
            new_grid.update_grid(cyl_sdfs, cyl_kidx, mode='overwrite')

            ## Step2: filter out grid points in the blended area
            vals1, common_kidx,_ = mc_grid.get_marked_intersection(cyl_kidx)
            vals2 = new_grid.get_vals(common_kidx)

            # NOTE: Area is: |d1-d2|_{n,delta} \leq delta
            # d1: value of handle cylinders implicit; 
            # d2: value of new part cylinder implicit
            area = smooth.abs(vals1 - vals2) <= delta
            area_kidx = common_kidx[area]

            mc_grid.clear_grid()
            new_grid.clear_grid()

        ## Step3: calculate SDF values of two shapes
        with torch.no_grad():
            for cid in range(handle.num_curve):
                curve = handle.curves[cid]
                key = self.encode_key(shape_name, curve.name)
                curve_data, kidx = curve.filter_grid(mc_grid)
                
                vals = self.__inference_vals(curve_data, key)
                # overwrite cylinder SDF, take min with other curve part
                mc_grid.update_grid_func(vals, kidx, func=np.minimum)

            key = self.encode_key(new_shape_name, new_curve.name)
            curve_data, new_kidx = new_curve.filter_grid(mc_grid)
            new_vals = self.__inference_vals(curve_data, key)
            new_grid.update_grid_func(new_vals, new_kidx, np.minimum)

            # cyl_sdfs, cyl_kidx = new_curve.calc_global_implicit(mc_grid, 0.)
            # pos = cyl_sdfs > -1.
            # pos_sdfs = cyl_sdfs[pos]
            # pos_kidx = cyl_kidx[pos]
            # new_grid.update_grid_func(pos_sdfs, pos_kidx, func=np.maximum)

        if arg['area_mode'] == 'small':
            # new_grid_kidx = np.argwhere(new_grid.func_marks).flatten()
            # area_marks = mc_grid.func_marks[new_grid_kidx]
            # area_kidx = new_grid_kidx[area_marks]

            # for mode-2: blending on intersection of cylinders
            area_marks = mc_grid.func_marks[new_kidx]
            area_kidx = new_kidx[area_marks]
        
        ## Step4: blend two shapes SDFs on the filtered grid points
        vals_shape = mc_grid.get_vals(area_kidx)
        vals_part = new_grid.get_vals(area_kidx)
        vals_area = smooth.min(vals_shape, vals_part)

        cyl_sdfs, cyl_kidx = new_curve.calc_global_implicit(mc_grid, delta)
        pos = cyl_sdfs > 0.
        pos_sdfs = cyl_sdfs[pos]
        pos_kidx = cyl_kidx[pos]
        mc_grid.update_grid(pos_sdfs, pos_kidx, mode='minimum')

        mc_grid.update_grid(new_vals, new_kidx, mode='minimum')
        mc_grid.update_grid(vals_area, area_kidx, mode='overwrite')

        mesh = mc_grid.extract_mesh()
        output_path = op.join(arg['output_path'], arg['config_name'])
        os.makedirs(output_path, exist_ok=True)
        out_name = '{}_{}|{}_{}.ply'.format(
            shape_name, new_shape_name, new_part_name, arg['exp_name']
        )
        # out_name = 'debug_blend.ply'
        mesh.export(op.join(output_path, out_name))
        print('{}|{} Done.'.format(
            arg['exp_name'], arg['config_name']
        ))

    def action_add_parts(self, arg):
        mc_grid = arg['mc_grid']
        delta = arg['delta']
        shape_arg = arg['shape']
        new_part_arg = arg['new_part']

        shape_name = shape_arg['name']
        smooth = utils.SmoothMaxMin(3, delta)
        new_grid = utils.create_grid_like(mc_grid)
        
        handle = self.handles[shape_name]
        for cid in range(handle.num_curve):
            curve = handle.curves[cid]
            key = self.encode_key(shape_name, curve.name)
            curve_data, kidx = curve.filter_grid(mc_grid)
            
            vals = self.__inference_vals(curve_data, key)
            # overwrite cylinder SDF, take min with other curve part
            mc_grid.update_grid_func(vals, kidx, np.minimum)

        for item_arg in new_part_arg:
            new_shape_name = item_arg['shape_name']
            new_part_name = item_arg['part_name']
            new_handle = self.handles[new_shape_name]
            if 'pose_file' in item_arg:
                pose_file = item_arg['pose_file']
                new_handle.apply_pose(pose_file)

            new_curve = new_handle.curve_dict[new_part_name]

            ## Step3: calculate SDF values of two shapes
            key = self.encode_key(new_shape_name, new_curve.name)
            curve_data, new_kidx = new_curve.filter_grid(mc_grid)
            new_vals = self.__inference_vals(curve_data, key)
            new_grid.update_grid_func(new_vals, new_kidx, np.minimum)

            # blending on intersection of cylinders
            area_marks = mc_grid.func_marks[new_kidx]
            area_kidx = new_kidx[area_marks]
        
            ## blend two shapes SDFs on the filtered grid points
            vals_shape = mc_grid.get_vals(area_kidx)
            vals_part = new_grid.get_vals(area_kidx)
            vals_area = smooth.min(vals_shape, vals_part)

            cyl_sdfs, cyl_kidx = new_curve.calc_global_implicit(mc_grid, 0.)
            pos = cyl_sdfs > 0.
            pos_sdfs = cyl_sdfs[pos]
            pos_kidx = cyl_kidx[pos]
            mc_grid.update_grid_func(pos_sdfs, pos_kidx, np.minimum)

            mc_grid.update_grid_func(new_vals, new_kidx, np.minimum)
            mc_grid.update_grid_func(vals_area, area_kidx, func=None)

            new_grid.clear_grid()

        mesh = mc_grid.extract_mesh()
        output_path = op.join(arg['output_path'], arg['config_name'])
        os.makedirs(output_path, exist_ok=True)
        out_name = '{}_{}.ply'.format(
            shape_name, arg['exp_name']
        )
        # out_name = 'debug_blend.ply'
        mesh.export(op.join(output_path, out_name))
        print('{}|{} Done.'.format(
            arg['exp_name'], arg['config_name']
        ))

    def action_slot_part(self, arg):
        mc_grid = arg['mc_grid']
        delta = arg['delta']
        shape_arg = arg['shape']
        shape_name = shape_arg['name']
        
        handle1 = self.handles[shape_name]

        new_part_arg = arg['new_part']
        new_shape_name = new_part_arg['shape_name']
        new_part_name = new_part_arg['part_name']
        handle2 = self.handles[new_shape_name]
        curve2_ori = handle2.curve_dict[new_part_name]
        curve2 = utils.copy_curve(handle2, new_part_name)
        curve2.apply_action_arg(new_part_arg)
        
        smooth = utils.SmoothMaxMin(3, delta)
        ball_arg = new_part_arg['ball']
        origin = ball_arg['origin']
        radius = ball_arg['radius']

        # shape1 sdf grid
        for cid in range(handle1.num_curve):
            curve = handle1.curves[cid]
            key = self.encode_key(shape_name, curve.name)
            curve_data, kidx = curve.filter_grid(mc_grid)
            
            vals = self.__inference_vals(curve_data, key)
            # overwrite cylinder SDF, take min with other curve part
            mc_grid.update_grid(vals, kidx, mode='minimum')

        # NOTE: filter radius + delta
        ball_pts, ball_kidx = mc_grid.filter_grid_ball(origin, radius+delta)
        sdf_val1 = mc_grid.get_vals(ball_kidx)
        sdf_ball = np.linalg.norm(ball_pts-origin, axis=1)
        sdf_ball -= radius
        sdf_ball = smooth.min(sdf_val1, sdf_ball)

        # move new part and re-scale
        anchor_idx = new_part_arg['anchor_idx']
        utils.curve_transform({
            'curve': curve2,
            'anchor_idx': anchor_idx,
            'origin': origin,
            'radius': radius,
        })

        # calculate extended part 
        area_coords, area_ts = curve2.core.localize_samples_global(ball_pts)
        points_shape2 = curve2_ori.core.inverse_transform(area_coords, area_ts)
        sdf_val2 = 10*np.ones(points_shape2.shape[0])
        for curve in handle2.curves:
            key = self.encode_key(new_shape_name, curve.name)
            curve_data, inside = curve.localize_samples(points_shape2)
            if np.any(inside):
                vals = self.__inference_vals(curve_data, key)
                # overwrite cylinder SDF, take min with other curve part
                vals = np.minimum(vals, sdf_val2[inside])
                sdf_val2[inside] = vals

        sdf_val2 = smooth.max(sdf_val2, sdf_ball)
        sdf_val1 = np.minimum(sdf_val1, sdf_val2)
        mc_grid.update_grid(sdf_val1, ball_kidx, mode='overwrite')

        key = self.encode_key(new_shape_name, curve2.name)
        curve_data, new_kidx = curve2.filter_grid(mc_grid)
        new_vals = self.__inference_vals(curve_data, key)

        # calculate intersection of ball and new part cylinder
        new_grid = utils.create_grid_like(mc_grid)
        new_grid.update_grid_func(sdf_val1, ball_kidx, func=None)
        area_marks = mc_grid.func_marks[new_kidx]
        area_kidx = new_kidx[area_marks]
        new_grid.update_grid_func(new_vals, new_kidx, func=smooth.min)
        area_vals = new_grid.get_vals(area_kidx)

        mc_grid.update_grid(new_vals, new_kidx, mode='minimum')
        mc_grid.update_grid(area_vals, area_kidx, mode='overwrite')
        mesh = mc_grid.extract_mesh()
        out_name = '{}_{}|{}_{}.ply'.format(
            shape_name, new_shape_name, new_part_name, arg['exp_name']
        )
        self.output_mesh(mesh, out_name, arg)
        

    def action_slot_move_part(self, arg):
        mc_grid = arg['mc_grid']
        delta = arg['delta']
        shape_arg = arg['shape']
        shape_name = shape_arg['name']
        part_name = shape_arg['part']
        
        handle = self.handles[shape_name]

        smooth = utils.SmoothMaxMin(3, delta)
        curve_ori = handle.curve_dict[part_name]
        curve_new = utils.copy_curve(handle, part_name)
        curve_new.apply_action_arg(shape_arg)

        for cid in range(handle.num_curve):
            curve = handle.curves[cid]
            if curve.name == part_name:
                continue

            key = self.encode_key(shape_name, curve.name)
            curve_data, kidx = curve.filter_grid(mc_grid)
            
            vals = self.__inference_vals(curve_data, key)
            # overwrite cylinder SDF, take min with other curve part
            mc_grid.update_grid_func(vals, kidx, func=np.minimum)

        anchor_idx = shape_arg['anchor_idx']
        origin = curve_new.core.key_points[anchor_idx]
        radius = curve_new.core.key_radius[anchor_idx].max()
        # NOTE: filter radius + delta
        ball_pts, ball_kidx = mc_grid.filter_grid_ball(origin, radius+delta)
        sdf_val1 = mc_grid.get_vals(ball_kidx)
        sdf_ball = np.linalg.norm(ball_pts-origin, axis=1)
        sdf_ball -= radius
        sdf_ball = smooth.min(sdf_val1, sdf_ball)

        # calculate extended part 
        area_coords, area_ts = curve_new.core.localize_samples_global(ball_pts)
        points_shape = curve_ori.core.inverse_transform(area_coords, area_ts)
        sdf_val2 = 10*np.ones(points_shape.shape[0])
        for curve in handle.curves:
            key = self.encode_key(shape_name, curve.name)
            curve_data, inside = curve.localize_samples(points_shape)
            if np.any(inside):
                vals = self.__inference_vals(curve_data, key)
                # overwrite cylinder SDF, take min with other curve part
                vals = np.minimum(vals, sdf_val2[inside])
                sdf_val2[inside] = vals
        
        sdf_val2 = smooth.max(sdf_val2, sdf_ball)
        sdf_val1 = np.minimum(sdf_val1, sdf_val2)
        mc_grid.update_grid(sdf_val1, ball_kidx, mode='overwrite')

        key = self.encode_key(shape_name, curve_new.name)
        curve_data, new_kidx = curve_new.filter_grid(mc_grid)
        new_vals = self.__inference_vals(curve_data, key)

        # calculate intersection of ball and new part cylinder
        new_grid = utils.create_grid_like(mc_grid)
        new_grid.update_grid_func(sdf_val1, ball_kidx, func=None)
        area_marks = mc_grid.func_marks[new_kidx]
        area_kidx = new_kidx[area_marks]
        new_grid.update_grid_func(new_vals, new_kidx, func=smooth.min)
        area_vals = new_grid.get_vals(area_kidx)

        mc_grid.update_grid(new_vals, new_kidx, mode='minimum')
        mc_grid.update_grid(area_vals, area_kidx, mode='overwrite')
        mesh = mc_grid.extract_mesh()
        out_name = '{}|{}_{}.ply'.format(
            shape_name, part_name, arg['exp_name']
        )
        self.output_mesh(mesh, out_name, arg)