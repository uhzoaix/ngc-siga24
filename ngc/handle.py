import os, pickle
import trimesh
import os.path as op
import numpy as np
from time import time
from handle_utils import EasyGraph, MCGrid
from scipy.spatial.transform import Rotation
from curve_handle import CurveHandle


class Handle():
    """docstring for Handle.
    """
    def __init__(self, arg=None):
        if arg is None:
            return 
        
        self.load_setting(arg)


    def compute_voxel_mask(self, reso, vertices):
        mc_config = {
            'reso': reso,
            'level': 0.,
        }
        mc_grid = MCGrid(mc_config)
        voxel = mc_grid.voxelize_points(vertices)
        mask = []
        voxel_parts = np.zeros((self.num_curve, voxel.shape[0]), dtype=bool)
        for cid, curve in enumerate(self.curves):
            _, kidx = curve.calc_cylinder_SDF(mc_grid)
            mask.append(kidx)
            voxel_parts[cid, kidx] = voxel[kidx]
        
        self.mask = mask
        N1 = mc_grid.reso + 1
        return voxel_parts.reshape((self.num_curve, N1, N1, N1))
    
    def precompute_inside_mask(self, points):
        inside_mat = np.zeros((self.num_curve, points.shape[0]))
        for cid,curve in enumerate(self.curves):
            vidx = curve.find_inside(points)
            inside_mat[cid, vidx] = 1.
        
        inside_mat /= np.sum(inside_mat, axis=1, keepdims=True)
        return inside_mat

    def filter_grid(self, mc_grid, curve_id):
        curve = self.curves[curve_id]
        return curve.filter_grid(mc_grid)
    
    def decompose_mesh(self, mesh):
        V = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        centers = np.mean(V[faces], axis=1)
        
        parts_fid = []
        names = []
        for curve in self.curves:
            fid = curve.find_inside(centers)
            parts_fid.append(fid)
            names.append(curve.name)

        parts = mesh.submesh(parts_fid)
        return parts, names
    
    def get_curve_scales(self, factor=0.9):
        scales = np.asarray([curve.get_bbox_scale() for curve in self.curves])
        return scales / factor

    def prepare_train_data(self, space_samples, surface_samples):
        res = []
        for curve in self.curves:
            name = curve.name
            part_surface = surface_samples[name]
            inbbox = curve.find_inbbox(space_samples)
            part_space = space_samples[inbbox]
            part_samples = np.concatenate([part_surface, part_space], axis=0)

            curve_data, _ = curve.localize_samples(part_samples)
            curve_data['name'] = curve.name

            res.append(curve_data)

        return res
    
    def prepare_samples(self, samples):
        # samples to train data
        samples_glob = []
        samples_local = []
        coords = []
        cids = []

        n_s = samples.shape[0]
        sidx = np.arange(n_s)

        for cid,curve in enumerate(self.curves):
            inbbox = curve.find_inbbox(samples)
            samples_bbox = samples[inbbox]
            sidx_bbox = sidx[inbbox]

            curve_data, inside = curve.localize_samples(samples_bbox)
            sidx_inside = sidx_bbox[inside]
            num_inside = sidx_inside.shape[0]

            samples_glob.append(curve_data['samples'])
            samples_local.append(curve_data['samples_local'])
            coords.append(curve_data['coords'])
            cids.append(np.ones(num_inside, dtype=int)*cid)


        samples_glob = np.concatenate(samples_glob, axis=0)
        samples_local = np.concatenate(samples_local, axis=0)
        coords = np.concatenate(coords)
        cids = np.concatenate(cids)

        return {
            'samples': samples_glob,
            'samples_local': samples_local,
            'coords': coords,
            'curve_idx': cids,
        }

    
    def prepare_occ(self, samples):
        samples_local = []
        coords = []
        for curve in self.curves:
            curve_data = curve.core.localize_occ_samples(samples)
            samples_local.append(curve_data['samples_local'])
            coords.append(curve_data['coords'])

        samples_local = np.stack(samples_local)
        coords = np.stack(coords)
        return {
            'samples_local': samples_local,
            'coords': coords,
        }


    def export_skeleton_mesh(self, output_path, reso=64, pose='std'):
        # handle skeleton
        skeleton = {}
        mc_config = {
            'reso': reso,
            'level': 0.
        }
        mc_grid = MCGrid(mc_config)
        
        for curve in self.curves:
            skeleton[curve.name] = curve.export_vis()
            sdfs, kidx = curve.calc_cylinder_SDF(mc_grid)
            mc_grid.update_grid(sdfs, kidx, mode='minimum')

        with open(op.join(output_path, f'{pose}_skeleton.pkl'), 'wb') as f:
            pickle.dump(skeleton, f)

        mesh = mc_grid.extract_mesh()
        mesh.export(op.join(output_path, f'{pose}_mesh.ply'))

    def generate_meshes(self, output_path, reso=64):
        mc_config = {
            'reso': reso,
            'level': 0.
        }
        mc_grid = MCGrid(mc_config)
        for curve in self.curves:
            sdfs, kidx = curve.calc_cylinder_SDF(mc_grid)
            mc_grid.update_grid(sdfs, kidx, mode='minimum')
            mesh = mc_grid.extract_mesh()
            mesh.export(op.join(output_path, f'curve_{curve.name}.ply'))
            mc_grid.clear_grid()

    def export_neural_graph(self):
        # export graph for neural learning
        verts = []
        edges = []
        rs = []
        num = 0
        for curve in self.curves:
            edges.append([num, num+1])
            verts.append(curve.core.key_points[0])
            verts.append(curve.core.key_points[-1])
            rs.append(curve.core.key_radius[0])
            rs.append(curve.core.key_radius[-1])
            
            num += 2

        return {
            'verts': np.asarray(verts),
            'radius': np.asarray(rs),
            'edges': np.asarray(edges),
        }

    def get_names(self):
        return [curve.name for curve in self.curves]

    def action_rotate(self, curve_name, vec, anchor_idx=0):
        curve = self.curve_dict[curve_name]
        x_axis = curve.core.key_frame[-1, 0]
        norm = np.linalg.norm(vec)
        vec = vec / norm
        quat = np.zeros(4)
        quat[:3] = np.cross(x_axis, vec)
        quat[3] = np.sqrt(np.sum(x_axis**2)* np.sum(vec**2)) + x_axis @ vec
        quat /= np.linalg.norm(quat)

        rot = Rotation.from_quat(quat)
        anchor = curve.core.key_points[anchor_idx]
        curve.apply_rotation(anchor, rot)
        curve.update()
        return rot
    
    def action_rotate_euler(self, arg):
        curve_name = arg['curve']
        anchor = arg['anchor']
        euler = arg['euler']
        curve = self.curve_dict[curve_name]
        rot = Rotation.from_euler(**euler)
        curve.apply_rotation(anchor, rot)
        curve.update()
        return rot
    
    
    def apply_pose(self, pose_file, z_axis=None):
        if isinstance(pose_file, str):
            pose_data = pickle.load(open(pose_file, 'rb'))
        else:
            pose_data = pose_file
            
        for name, sk in pose_data.items():
            curve = self.curve_dict[name]
            curve.set_resamples(sk['vertices'], z_axis)
            curve.update()
    
    def apply_scaling(self, arg):
        for name, val in arg.items():
            curve = self.curve_dict[name]
            scales = np.asarray(val['scales'])
            ts = np.asarray(val['coords'])
            curve.radius_scaling(scales, ts)
            curve.update()
    
    def apply_tilt(self, arg):
        for name, val in arg.items():
            curve = self.curve_dict[name]
            angles = np.asarray(val['angles'])* (np.pi/ 180)
            ts = np.asarray(val['coords'])
            curve.rot_tilt(angles, ts)
            curve.update()
    
    def calc_bbox(self):
        bmaxs = []
        bmins = []
        for curve in self.curves:
            bmax, bmin = curve.cyl_mesh.calc_bbox()
            bmaxs.append(bmax)
            bmins.append(bmin)
        
        bmaxs = np.stack(bmaxs)
        bmins = np.stack(bmins)
        res_bmax = np.max(bmaxs, axis=0)
        res_bmin = np.min(bmins, axis=0)
        res_bmax = np.clip(res_bmax, a_min=-1., a_max=1.)
        res_bmin = np.clip(res_bmin, a_min=-1., a_max=1.)
        return res_bmax, res_bmin

    def print_info(self):
        for cid, curve in enumerate(self.curves):
            print(f'Curve{cid}: {curve.name}')
            curve.print_info()

    def set_node_idx(self, name, edge):
        if not hasattr(self, 'node_idx'):
            self.node_idx = {}
        
        self.node_idx[name] = edge

    def get_node_idx(self, name):
        return self.node_idx[name]

    def gen_cyl_mesh(self, output_path):
        for curve in self.curves:
            cyl_mesh = curve.cyl_mesh.extract_mesh()
            mesh_path = op.join(output_path, f'{curve.name}_cyl.ply')
            cyl_mesh.export(mesh_path)
        print('Done')

    def get_info(self, handle_path):
        # e.g. for dataloading of dataset in training
        # get useful info without fully loading
        with open(handle_path, 'rb') as f:
            data = pickle.load(f)

        num_curve = len(data['curves'])
        
        info = {
            'num_curve': num_curve,
            'curve_idx': list(range(num_curve)),
        }
        return info
    

    def load_setting(self, data):
        cid = 0
        self.curves = []
        self.curve_dict = {}
        for name, cdata in data.items():
            ball = cdata['ball'] if 'ball' in cdata else None
            curve_arg = {
                'name': name,
                'idx': cid,
                'ball': ball,
                'z_axis': cdata['z_axis'],
                'key_radius': cdata['radius'],
                'key_points': cdata['points'],
            }
            curve = CurveHandle(curve_arg)
            self.curve_dict[name] = curve
            self.curves.append(curve)
            cid += 1

        self.num_curve = len(self.curves)
        
    def load(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.curves = []
        self.curve_dict = {}
        for cid, curve_data in enumerate(data['curves']):
            curve_data['idx'] = cid
            curve = CurveHandle()
            curve.load_data(curve_data)
            self.curve_dict[curve.name] = curve
            self.curves.append(curve)

        self.num_curve = len(self.curves)


    def export(self, output_file):
        data = {}
        data['curves'] = [curve.export_data() for curve in self.curves]

        with open(output_file, 'wb') as f:
            pickle.dump(data, f)


def Handle_from_setting():
    item = 'stickman'
    root_path = f'/data/NGCDataset/Pack50Dataset/{item}'
    handle_path = op.join(root_path, 'handle_setting.pkl')
    handle = Handle()
    handle_setting = pickle.load(open(handle_path, 'rb'))
    handle.load_setting(handle_setting)

    output_path = op.join(root_path, 'handle')
    os.makedirs(output_path, exist_ok=True)
    handle.export_skeleton_mesh(output_path)
    handle.export(op.join(output_path, 'std_handle.pkl'))
    print(f'{item} handle processed')


def process_handle():
    root_path = f'/data/NGCDataset/Pack50Dataset'
    item_file = op.join(root_path, 'data.txt')
    items = np.loadtxt(item_file, dtype=str)    
    for item in items:
        item_path = op.join(root_path, item)
        handle_path = op.join(item_path, 'handle', 'std_handle.pkl')
        handle = Handle()
        handle.load(handle_path)

        handle.export_skeleton_mesh(op.join(item_path, 'handle'), reso=256)

    print('Done')

if __name__ == "__main__":
    # Handle_from_setting()
    process_handle()