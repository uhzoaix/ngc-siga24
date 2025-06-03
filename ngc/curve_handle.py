import os, pickle
import numpy as np
import os.path as op
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation, Slerp
from handle_utils import CylindersMesh


class CurveHandle():
    """docstring for CurveHandle."""
    def __init__(self, arg=None):
        if arg is None:
            return
        
        self.set_curve(arg)

    def update(self):
        self.core.update()
        n_sample_curve = 20
        n_sample_circle = 12
        self.cyl_mesh = self.gen_cylinder_mesh(n_sample_curve, n_sample_circle)

    def set_curve(self, arg):
        self.name = arg['name']
        self.idx = arg['idx']
        if not hasattr(self, 'core'):
            self.core = PWLACurve(arg)
        else:
            self.core.set_curve(arg)
        
        self.update()

    def set_points(self, points):
        # number of points remain same
        self.core.set_points(points)

    def set_resamples(self, points, z_axis0=None):
        self.core.set_resamples(points, z_axis0)

    def apply_action_arg(self, arg):
        if 'pose_file' in arg:
            pose_file = arg['pose_file']
            pose_data = pickle.load(open(pose_file, 'rb'))
            sk = pose_data[self.name]
            self.set_resamples(sk['vertices'])
            self.update()

        if 'rotation' in arg:
            rot_arg = arg['rotation']
            vec = rot_arg['vec']
            anchor_idx = rot_arg['anchor_idx']
            x_axis = self.core.key_frame[-1, 0]
            norm = np.linalg.norm(vec)
            vec = vec / norm
            quat = np.zeros(4)
            quat[:3] = np.cross(x_axis, vec)
            quat[3] = np.sqrt(np.sum(x_axis**2)* np.sum(vec**2)) + x_axis @ vec
            quat /= np.linalg.norm(quat)

            rot = Rotation.from_quat(quat)
            anchor = self.core.key_points[anchor_idx]
            self.apply_rotation(anchor, rot)
            self.update()

    def apply_rotation(self, anchor, rot):
        self.core.apply_rotation(anchor, rot)

    def radius_scaling(self, scales, coords):
        key_radius_scales = np.interp(
            self.core.key_ts, coords, scales
        )
        self.core.key_radius *= key_radius_scales[:,None]

    def rot_part(self, idx, axis, angle):
        anchor = self.core.key_points[idx]
        part_pts = self.core.key_points[idx:]
        part_zaxis = self.core.z_axis[idx:]
        rot = Rotation.from_euler(axis, angle, degrees=True)
        new_pts = rot.apply(part_pts - anchor) + anchor
        new_zaxis = rot.apply(part_zaxis)

        self.core.key_points[idx:] = new_pts
        self.core.z_axis[idx:] = new_zaxis
        self.core.flag_points = True
        return anchor, rot
    
    def stretch_part(self, front, back):
        idx1, offset1 = front
        idx2, offset2 = back

        self.core.key_points[idx1:] += offset1
        self.core.key_points[:(idx2+1)] += offset2
        # NOTE: do not update, since it will recalculate natural coords
        n_sample_curve = 20
        n_sample_circle = 12
        self.cyl_mesh = self.gen_cylinder_mesh(n_sample_curve, n_sample_circle)

    def rot_tilt(self, angles, coords):
        key_angles = np.interp(
            self.core.key_ts, coords, angles
        )
        a11 = np.sin(key_angles + np.pi/2)[:,None]
        a12 = np.cos(key_angles + np.pi/2)[:,None]
        a21 = np.sin(key_angles)[:,None]
        a22 = np.cos(key_angles)[:,None]
        
        kf = self.core.key_frame.copy()
        new_y = kf[:,1,:]*a11 + kf[:,2,:]*a12
        new_z = kf[:,1,:]*a21 + kf[:,2,:]*a22
        kf[:,1,:] = new_y
        kf[:,2,:] = new_z
        self.core.set_frame(kf)


    def apply_translation(self, offset):
        self.core.key_points += offset

    def clip_cylinder(self, t0=None, t1=None):
        if t0 is not None:
            idx0 = np.searchsorted(self.core.key_ts, t0)
            itp0 = self.core.interpolate([t0])
            p0,r0,f0 = itp0['points'], itp0['radius'], itp0['frame']
            key_ts = np.concatenate([[t0], self.core.key_ts[idx0+1:]])
            key_points = np.concatenate([p0, self.core.key_points[idx0+1:]], axis=0)
            key_radius = np.concatenate([r0, self.core.key_radius[idx0+1:]], axis=0)
            key_frame = np.concatenate([f0, self.core.key_frame[idx0+1:]], axis=0)
            self.core.key_ts = key_ts
            self.core.key_points = key_points
            self.core.key_radius = key_radius
            self.core.key_frame = key_frame
        
        if t1 is not None:
            idx1 = np.searchsorted(self.core.key_ts, t1)
            idx1 = min(idx1, self.core.key_ts.shape[0]-1)
            itp1 = self.core.interpolate([t1])
            p1,r1,f1 = itp1['points'], itp1['radius'], itp1['frame']
            key_ts = np.concatenate([self.core.key_ts[:idx1], [t1]])
            key_points = np.concatenate([self.core.key_points[:idx1], p1], axis=0)
            key_radius = np.concatenate([self.core.key_radius[:idx1], r1], axis=0)
            key_frame = np.concatenate([self.core.key_frame[:idx1], f1], axis=0)
            self.core.key_ts = key_ts
            self.core.key_points = key_points
            self.core.key_radius = key_radius
            self.core.key_frame = key_frame

        self.update()

    def get_end_data(self, side):
        return {
            'point': self.core.key_points[side],
            'radius': self.core.key_radius[side],
            'frame': self.core.key_frame[side],
        }

    def gen_cylinder_mesh(self, n_sample_curve, n_sample_circle):
        # t0,t1 = [0., 1.]
        t0 = self.core.key_ts[0]
        t1 = self.core.key_ts[-1]
        ts = np.linspace(t0, t1, n_sample_curve)
        thetas = (2*np.pi)* np.linspace(0, 1, n_sample_circle, endpoint=False)

        intpl = self.core.interpolate(ts)
        intpl['thetas'] = thetas
        return self.__gen_cyl_mesh(intpl)

    def __gen_cyl_mesh(self, intpl):
        thetas = intpl['thetas']
        verts = intpl['points']
        yz_rs = intpl['radius']
        frames = intpl['frame']
        if 'level' in intpl:
            level = intpl['level']
        else:
            level = 0.

        theta_coo = np.asarray([np.cos(thetas), np.sin(thetas)])
        theta_coo = np.einsum('nj,ji->nji', yz_rs, theta_coo)
        cyl_pts = np.einsum('njk,nji->nik', frames[:,1:], theta_coo)
        
        # points of level set
        dists = np.linalg.norm(cyl_pts, axis=2)
        scales = (dists + level) / dists
        cyl_pts *= scales[..., None]

        cyl_pts += verts[:, None, :]

        v0, v1 = verts[0], verts[-1]
        if self.core.use_ball('start'):
            # for level set
            disp = self.core.ball_disp('start', level)

            start_pts = cyl_pts[0] + disp
            cyl_pts = np.concatenate([
                start_pts[None, ...], cyl_pts], axis=0)
            
            v0 += disp
        
        if self.core.use_ball('end'):
            # for level set
            disp = self.core.ball_disp('end', level)

            end_pts = cyl_pts[-1] + disp
            cyl_pts = np.concatenate([
                cyl_pts, end_pts[None, ...]], axis=0)
            v1 += disp

        cyl_mesh = CylindersMesh()
        vidx = cyl_mesh.add_cylinder(cyl_pts)

        cyl_mesh.add_cap(v0, vidx[0])
        cyl_mesh.add_cap(v1, vidx[-1], flip_face=True)
        return cyl_mesh

    def generate_samples(self, num_samples):
        return self.core.generate_samples(num_samples)

    def localize_samples(self, samples):
        return self.core.localize_samples(samples)
    
    # def localize_samples_transition(self, samples, ts):
    #     return self.core.localize_samples_transition(samples, ts)

    def filter_grid(self, mc_grid):
        # find grid points around the cylinder
        samples, kidx = self.cyl_mesh.filter_grid(mc_grid)
        # calculate the neareast points and find inside points
        samples_data, inside = self.core.localize_samples(samples)
        kidx = kidx[inside]
        return samples_data, kidx
    
    def filter_grid_mix(self, mc_grid, mix_arg):
        # gen mixed cyl mesh 
        ts = np.linspace(0., 1., 20)
        
        intpl = self.core.interpolate_mix(ts, mix_arg)
        thetas = (2*np.pi)* np.linspace(0, 1, 12, endpoint=False)
        intpl['thetas'] = thetas
        cyl_mesh = self.__gen_cyl_mesh(intpl)
        
        samples, kidx = cyl_mesh.filter_grid(mc_grid)
        samples_data, inside = self.core.localize_samples_mix(samples, mix_arg)
        kidx = kidx[inside]
        return samples_data, kidx

    def filter_grid_stretch(self, mc_grid, blend_arg):
        # gen mixed cyl mesh 
        ts = np.linspace(0., 1., 20)
        
        intpl,_ = self.core.interpolate_stretch(ts, blend_arg)
        thetas = (2*np.pi)* np.linspace(0, 1, 12, endpoint=False)
        intpl['thetas'] = thetas
        cyl_mesh = self.__gen_cyl_mesh(intpl)
        
        samples, kidx = cyl_mesh.filter_grid(mc_grid)
        samples_data, inside = self.core.localize_samples_stretch(samples, blend_arg)
        kidx = kidx[inside]
        return samples_data, kidx

    def calc_cylinder_SDF(self, mc_grid):
        samples, kidx = self.cyl_mesh.filter_grid(mc_grid, extend_bbox=True)
        # NOTE: value of samples_sdf is calculated in std space
        samples_sdf = self.core.calc_cylinder_SDF(samples)
        return samples_sdf, kidx
    
    def calc_global_implicit(self, mc_grid, sigma, return_coords=False):
        # gen blended cyl mesh 
        ts = np.linspace(0., 1., 20)
        
        intpl = self.core.interpolate(ts)
        thetas = (2*np.pi)* np.linspace(0, 1, 12, endpoint=False)
        intpl['thetas'] = thetas
        intpl['level'] = sigma
        cyl_mesh = self.__gen_cyl_mesh(intpl)
        
        samples, kidx = cyl_mesh.filter_grid(mc_grid)
        if not return_coords:
            sdfs = self.core.calc_global_implicit(samples)
            return sdfs, samples, kidx
        else:
            sdfs, coords, ts = self.core.calc_global_implicit(samples, True)
            return {
                'sdf': sdfs,
                'kidx': kidx,
                'coords': coords,
                'ts': ts,
            }
        # neg = sdfs <= 0.
        # return samples[neg]
   
    
    def get_bbox_scale(self):
        bmax, bmin = self.cyl_mesh.calc_bbox()
        scale = np.max(bmax-bmin)/ 2.
        return scale
    
    def find_inbbox(self, points):
        bmax, bmin = self.cyl_mesh.calc_bbox()
        side1 = np.all((points - bmin) >= 0, axis=1)
        side2 = np.all((bmax - points) >= 0, axis=1)
        inside = np.logical_and(side1, side2)
        return inside

    def find_inside(self, points):
        pidx = np.arange(points.shape[0])
        inbbox = self.find_inbbox(points)
        pidx_bbox = pidx[inbbox]

        _, inside = self.core.localize_samples(points[inbbox])
        pidx_inside = pidx_bbox[inside]
        return pidx_inside
    
    def print_info(self):
        # print('Points: ', self.core.key_points)
        # print('Radius: ', self.core.key_radius)
        print('Start radius: ', self.core.key_radius[0])
        print('Start x radius: ', self.core.start_ball_x)
        
    def load_data(self, arg):
        self.set_curve(arg)

    def export_data(self):
        data = self.core.export_data()
        data['name'] = self.name
        return data
    
    def export_vis(self):
        return self.core.export_vis()


class PWLACurve():
    """docstring for PWLACurve."""
    def __init__(self, arg=None):
        if arg is None:
            return 
        
        self.set_curve(arg)

    def update(self):
        if self.flag_points:
            # if some keypoints changed, update the curve
            self.update_coords()
            self.update_frame()

            self.flag_points = False
            return
        
    def need_update(self):
        return self.flag_points
    
    def use_ball(self, side):
        return getattr(self, f'{side}_ball_x') is not None
    
    def ball_disp(self, side, level=0.):
        if side == 'start':
            tan = self.key_points[1] - self.key_points[0]
            tan /= np.linalg.norm(tan)
            disp = self.start_ball_x + level
            return -disp* tan
        
        if side == 'end':
            tan = self.key_points[-1] - self.key_points[-2]
            tan /= np.linalg.norm(tan)
            disp = self.end_ball_x + level
            return disp* tan

    def set_curve(self, arg):
        # self.step = arg['resample_step']
        self.key_points = arg['key_points']
        # NOTE: radius: (N, 2), y-z radius
        self.key_radius = arg['key_radius']
        # self.end_ball_x = self.key_radius[-1].mean()
        # self.start_ball_x = self.key_radius[0].mean()
        # self.end_ball_x = None
        # self.start_ball_x = None
        ball_data = arg['ball']
        if ball_data is None:
            self.end_ball_x = None
            self.start_ball_x = None
        else:
            if ball_data['end_x'] is not None:
                self.end_ball_x = ball_data['end_x']
            else:
                self.end_ball_x = None

            if ball_data['start_x'] is not None:
                self.start_ball_x = ball_data['start_x']
            else:
                self.start_ball_x = None

        x_axis = self.estimate_tangent(self.key_points)
        z_axis = arg['z_axis']
        if len(z_axis.shape) == 1:
            z_axis = np.tile(z_axis, (x_axis.shape[0], 1))
        
        self.z_axis = self.project_z_axis(x_axis, z_axis)

        self.update_coords()
        # check if points or radius have to be updated
        self.flag_points = True
        # self.flag_radius = False

    def calc_curve_length(self):
        pts = self.key_points
        edge_vec = pts[1:] - pts[:-1]
        edge_lengths = np.linalg.norm(edge_vec, axis=1)
        curve_length = np.sum(edge_lengths)
        return curve_length

    def set_points(self, points):
        assert (points.shape == self.key_points.shape)
        self.key_points = points
        self.flag_points = True

    def set_resamples(self, points, z_axis0):
        if z_axis0 is None:
            z_axis0 = self.z_axis[0]
        # new points can be different in number
        edge_vec = points[1:] - points[:-1]
        edge_lengths = np.linalg.norm(edge_vec, axis=1)
        curve_length = np.sum(edge_lengths)
        ts = np.cumsum(np.r_[0., edge_lengths]) / curve_length

        new_rs = self.interpolate(ts, radius=True)['radius']
        self.key_points = points
        self.key_ts = ts
        self.key_radius = new_rs
        x_axis = self.estimate_tangent(self.key_points)
        self.z_axis = self.propagate_z_axis(x_axis, z_axis0)
        y_axis = np.cross(self.z_axis, x_axis)
        self.key_frame = np.concatenate([
            x_axis.reshape(-1,1,3),
            y_axis.reshape(-1,1,3),
            self.z_axis.reshape(-1,1,3)
        ], axis=1)
        self.rotation = Rotation.from_matrix(self.key_frame)
        self.rot_slerp = Slerp(self.key_ts, self.rotation)
        self.flag_points = True


    def apply_rotation(self, anchor, rot):
        self.key_points = rot.apply(self.key_points - anchor) + anchor
        self.z_axis = rot.apply(self.z_axis)
        self.flag_points = True

    def estimate_tangent(self, points):
        edge_vec = points[1:] - points[:-1]
        edge_vec /= np.linalg.norm(edge_vec, axis=1, keepdims=True)

        if edge_vec.shape[0] > 1:
            tan_start = edge_vec[0]
            tan_end = edge_vec[-1]
            tans = (edge_vec[1:] + edge_vec[:-1]) / 2.
            tans /= np.linalg.norm(tans, axis=1, keepdims=True)

            vert_tan = np.concatenate([
                tan_start.reshape(1,3), 
                tans, 
                tan_end.reshape(1,3)
            ], axis=0)
        else:
            vert_tan = np.tile(edge_vec, (2,1))

        return vert_tan
    
    def project_z_axis(self, x_axis, z_axis):
        dots = np.sum(x_axis*z_axis, axis=1)
        if not np.allclose(dots, 0):
            # project z_axis to x_axis
            z_axis = z_axis - dots[:, None]* x_axis
            # NOTE: huh???? forgot this
            z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)

        return z_axis
    
    def propagate_z_axis(self, x_axis, z_axis0):
        final_z = []
        # current z_axis
        c_zx = z_axis0 
        for i in range(x_axis.shape[0]):
            xx = x_axis[i]
            zx = c_zx - (xx @ c_zx)*xx
            zx /= np.linalg.norm(zx)
            final_z.append(zx)
            c_zx = zx

        return np.asarray(final_z)
    
    def update_frame(self):
        x_axis = self.estimate_tangent(self.key_points)
        self.z_axis = self.project_z_axis(x_axis, self.z_axis)
        y_axis = np.cross(self.z_axis, x_axis)
        self.key_frame = np.concatenate([
            x_axis.reshape(-1,1,3),
            y_axis.reshape(-1,1,3),
            self.z_axis.reshape(-1,1,3)
        ], axis=1)
        self.rotation = Rotation.from_matrix(self.key_frame)
        self.rot_slerp = Slerp(self.key_ts, self.rotation)

    def set_frame(self, new_frame):
        self.key_frame = new_frame
        self.rotation = Rotation.from_matrix(self.key_frame)
        self.rot_slerp = Slerp(self.key_ts, self.rotation)

    def update_coords(self):
        edge_vec = self.key_points[1:] - self.key_points[:-1]
        edge_lengths = np.linalg.norm(edge_vec, axis=1)
        self.curve_length = np.sum(edge_lengths)
        self.key_ts = np.cumsum(np.r_[0., edge_lengths]) / self.curve_length


    def is_points_in_edge(self, points, vt0, vt1):
        # project points on line segment(edge), return if inside the edge 
        v0, t0 = vt0
        v1, t1 = vt1

        length = np.linalg.norm(v1 - v0)
        vec = (v1 - v0) / length
        proj_len = (points - v0) @ vec
        inside_flag = np.logical_and(proj_len >= 0., proj_len <= length)

        # calculate natural coords for projected points
        ts = t0 + ((t1 - t0)/length)* proj_len
        return inside_flag, ts
    

    def curve_projection(self, samples, N_discrete=20, outside=False):
        td = np.linspace(0., 1., N_discrete, endpoint=False)
        ii = np.searchsorted(td, self.key_ts)
        td = np.insert(td, ii, self.key_ts)
        td = np.unique(td)

        verts = self.interpolate(td, radius=False, frame=False)['points']
        tree = KDTree(verts)
        # not accurate for radius-varying skeleton
        _, vidx = tree.query(samples)
        ts = -1*np.ones(samples.shape[0])
        # basically project samples onto the piecewise linear curve
        num_vert = verts.shape[0]
        for vid in range(num_vert):
            sidx = np.argwhere(vidx == vid).flatten()
            if len(sidx) == 0:
                continue

            samples_v = samples[sidx]

            if 0 < vid < num_vert - 1:
                # middle part
                ts[sidx] = td[vid]

                in1, px1 = self.is_points_in_edge(
                    samples_v, 
                    (verts[vid], td[vid]), 
                    (verts[vid+1], td[vid+1])
                )
                in2, px2 = self.is_points_in_edge(
                    samples_v, 
                    (verts[vid-1], td[vid-1]), 
                    (verts[vid], td[vid])
                )
                in_p = np.logical_xor(in1, in2)
                px = (in1*px1 + in2*px2)[in_p]
                ts[sidx[in_p]] = px

            elif vid == 0:
                in1, px1 = self.is_points_in_edge(
                    samples_v, 
                    (verts[vid], td[vid]), 
                    (verts[vid+1], td[vid+1])
                )
                # consider halfball+ cylinder
                # left side of cylinder remain valid
                if self.start_ball_x is not None or outside:
                    ts[sidx] = 0.

                ts[sidx[in1]] = px1[in1]

            else:
                in2, px2 = self.is_points_in_edge(
                    samples_v, 
                    (verts[vid-1], td[vid-1]), 
                    (verts[vid], td[vid]), 
                )
                if self.end_ball_x is not None or outside:
                    ts[sidx] = 1.

                ts[sidx[in2]] = px2[in2]

        return ts

    def calc_x_radius(self, ts):
        xrs = np.ones(ts.shape[0])
        if self.end_ball_x is not None:
            xrs[ts == 1.] = self.end_ball_x
        
        if self.start_ball_x is not None:
            xrs[ts == 0.] = self.start_ball_x
        
        return xrs
    
    def calc_cylinder_SDF(self, vs):
        ts = self.curve_projection(vs, outside=True)

        intpl = self.interpolate(ts)
        proj_vs = intpl['points']
        yz_rs = intpl['radius']
        frame_mat = intpl['frame']

        x_rs = self.calc_x_radius(ts)
        radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)

        # frame: (N, 3,3), vs (N, 3)
        samples_local = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
        samples_local /= radius
        norms_cyl = np.linalg.norm(samples_local, axis=1)
        
        vx = 2*ts - 1
        samples_local[:, 0] += vx
        xpos = vx >= 0.
        xneg = np.logical_not(xpos)

        norms_max = np.linalg.norm(samples_local, axis=1, ord=np.inf)
        if self.start_ball_x is None:
            norms_cyl[xneg] = np.maximum(norms_cyl[xneg], norms_max[xneg])

        if self.end_ball_x is None:
            norms_cyl[xpos] = np.maximum(norms_cyl[xpos], norms_max[xpos])
            
        return norms_cyl - 1.
    
    def calc_std_cylinder_SDF(self, vs):
        # assume this is a standard cylinder
        # vs all inside the cylinder
        ts = self.curve_projection(vs)
        intpl = self.interpolate(ts)
        proj_vs = intpl['points']
        yz_rs = intpl['radius']
        
        dist_cyl = np.linalg.norm(vs - proj_vs, axis=1)
        dist_cyl = yz_rs[:,0] - dist_cyl

        tsd = np.minimum(ts, 1-ts)
        curve_length = self.calc_curve_length()
        dist_side = tsd*curve_length

        dist = np.minimum(dist_cyl, dist_side)
        return -dist
    
    def calc_global_implicit(self, vs, return_coords=False):
        ts = self.curve_projection(vs, outside=True)

        intpl = self.interpolate(ts)
        proj_vs = intpl['points']
        yz_rs = intpl['radius']
        frame_mat = intpl['frame']

        x_rs = self.calc_x_radius(ts)
        radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)

        # frame: (N, 3,3), vs (N, 3)
        samples_local = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
        samples_local /= radius
        norms_cyl = np.linalg.norm(samples_local, axis=1)
        # return norms_cyl - 1.
        signs = np.sign(norms_cyl - 1.)
        
        nearest_local = samples_local / norms_cyl[:, None]
        nearest_global = nearest_local* radius
        # NOTE: here it should be the inverse of frames
        # so it is the transpose(since they are unitary mats)
        # and we simply modify it in einsum: nij,nj->ni => nji,nj->ni
        nearest_global = np.einsum('nji,nj->ni', frame_mat, nearest_global)
        nearest_global += proj_vs
        ds = np.linalg.norm(nearest_global - vs, axis=1)
        if return_coords:
            vx = 2*ts - 1
            samples_local[:, 0] += vx
            return ds*signs, samples_local, ts
        else:
            return ds*signs

    def localize_samples(self, vs, return_sdf=False):
        ts = self.curve_projection(vs)
        ts_range = np.logical_and(ts >= 0., ts <= 1.)
        sidx = np.arange(vs.shape[0])
        ts = ts[ts_range]
        vs = vs[ts_range]
        sidx = sidx[ts_range]
        
        intpl = self.interpolate(ts)
        proj_vs = intpl['points']
        yz_rs = intpl['radius']
        frame_mat = intpl['frame']

        x_rs = self.calc_x_radius(ts)
        radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)

        # frame: (N, 3,3), vs (N, 3)
        samples_local = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
        samples_local /= radius

        # in std cylinder
        norms = np.linalg.norm(samples_local, axis=1)
        if return_sdf:
            return norms - 1, sidx
        
        inside_cyl = norms <= 1
        inside = sidx[inside_cyl]

        # NOTE: vs -> (vx, *, *). (vert -> (vx, 0, 0))
        # [0,1] -> [-1,1]
        vx = 2*ts - 1
        samples_local[:, 0] += vx
        return {
            'samples': vs[inside_cyl],
            'samples_local': samples_local[inside_cyl],
            'coords': ts[inside_cyl],
            # 'radius': yz_rs[inside_cyl],
        }, inside
    
    def localize_samples_global(self, vs):
        ts = self.curve_projection(vs, outside=True)

        intpl = self.interpolate(ts)
        proj_vs = intpl['points']
        yz_rs = intpl['radius']
        frame_mat = intpl['frame']

        x_rs = self.calc_x_radius(ts)
        radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)

        # frame: (N, 3,3), vs (N, 3)
        samples_local = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
        samples_local /= radius
        vx = 2*ts - 1
        samples_local[:, 0] += vx
        return samples_local, ts


    def localize_samples_mix(self, vs, mix_arg):
        ts = self.curve_projection(vs)
        ts_range = np.logical_and(ts >= 0., ts <= 1.)
        sidx = np.arange(vs.shape[0])
        ts = ts[ts_range]
        vs = vs[ts_range]
        sidx = sidx[ts_range]
        
        intpl = self.interpolate_mix(ts, mix_arg)
        proj_vs = intpl['points']
        yz_rs = intpl['radius']
        frame_mat = intpl['frame']

        x_rs = self.calc_x_radius(ts)
        radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)

        # frame: (N, 3,3), vs (N, 3)
        samples_local = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
        samples_local /= radius

        # in std cylinder
        norms = np.linalg.norm(samples_local, axis=1)
        inside_cyl = norms <= 1
        inside = sidx[inside_cyl]

        # NOTE: vs -> (vx, *, *). (vert -> (vx, 0, 0))
        # [0,1] -> [-1,1]
        vx = 2*ts - 1
        samples_local[:, 0] += vx
        return {
            'samples': vs[inside_cyl],
            'samples_local': samples_local[inside_cyl],
            'coords': ts[inside_cyl],
            # 'radius': yz_rs[inside_cyl],
        }, inside

    def localize_samples_stretch(self, vs, blend_arg):
        # for stretch or offset 
        ts = self.curve_projection(vs)
        ts_range = np.logical_and(ts >= 0., ts <= 1.)
        sidx = np.arange(vs.shape[0])
        ts = ts[ts_range]
        vs = vs[ts_range]
        sidx = sidx[ts_range]
        
        intpl, ts_new = self.interpolate_stretch(ts, blend_arg)
        proj_vs = intpl['points']
        frame_mat = intpl['frame']
        # for yz radius use ts_new(stretch or offset)
        yz_rs = intpl['radius']

        x_rs = self.calc_x_radius(ts)
        radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)

        # frame: (N, 3,3), vs (N, 3)
        samples_local = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
        samples_local /= radius

        # in std cylinder
        norms = np.linalg.norm(samples_local, axis=1)
        inside_cyl = norms <= 1
        inside = sidx[inside_cyl]

        # NOTE: vs -> (vx, *, *). (vert -> (vx, 0, 0))
        # [0,1] -> [-1,1]
        vx = 2*ts_new - 1
        samples_local[:, 0] += vx
        return {
            'samples': vs[inside_cyl],
            'samples_local': samples_local[inside_cyl],
            'coords': ts_new[inside_cyl],
            # 'radius': yz_rs[inside_cyl],
        }, inside


    def localize_occ_samples(self, samples):
        ts = self.curve_projection(samples, outside=True)

        intpl = self.interpolate(ts)
        proj_vs = intpl['points']
        frame_mat = intpl['frame']

        # frame: (N, 3,3), vs (N, 3)
        samples_local = np.einsum('nij,nj->ni', frame_mat, (samples - proj_vs))

        vx = 2*ts - 1
        samples_local[:, 0] += vx
        return {
            'samples_local': samples_local,
            'coords': ts
        }


    def interpolate(self, ts, points=True, radius=True, frame=True):
        res = {}
        # key_points: (N, 3); key_radius: (N, 2)
        if points:
            pts_ts = np.stack([
                np.interp(ts, self.key_ts, self.key_points[:, 0]),
                np.interp(ts, self.key_ts, self.key_points[:, 1]),
                np.interp(ts, self.key_ts, self.key_points[:, 2])
            ]).T
            res['points'] = pts_ts
        
        if radius:
            rs_ts = np.stack([
                np.interp(ts, self.key_ts, self.key_radius[:, 0]),
                np.interp(ts, self.key_ts, self.key_radius[:, 1])
            ]).T
            res['radius'] = rs_ts

        if frame:
            # requirement of Slerp from Scipy
            ts_rot = np.clip(ts, a_min=1e-10, a_max=(1 - 1e-10))
            frame_ts = self.rot_slerp(ts_rot).as_matrix()
            res['frame'] = frame_ts

        return res

    def interpolate_mix(self, ts, mix_arg):
        new_curve = mix_arg['curve_handle']
        func1 = mix_arg['mix_func1']
        func2 = mix_arg['mix_func2']
        ts1, weights1 = func1(ts)
        ts2, weights2 = func2(ts)

        rs1 = np.stack([
            np.interp(ts1, self.key_ts, self.key_radius[:, 0]),
            np.interp(ts1, self.key_ts, self.key_radius[:, 1])
        ]).T
        rs1_mean = rs1.mean(axis=1)
        scales1 = rs1 / rs1_mean[:, None]

        intpl2 = new_curve.core.interpolate(ts2, points=False, frame=False)
        rs2 = intpl2['radius']
        rs2_mean = rs2.mean(axis=1)
        scales2 = rs2 / rs2_mean[:, None]

        scales = scales1*weights1[:,None] + scales2*weights2[:,None]
        radius = rs1_mean[:,None]*scales

        intpl = self.interpolate(ts, radius=False)
        intpl['radius'] = radius
        return intpl
    
    def interpolate_stretch(self, ts, mix_arg):
        func = mix_arg['mix_func']
        ts_new = func(ts)

        rs1 = np.stack([
            np.interp(ts_new, self.key_ts, self.key_radius[:, 0]),
            np.interp(ts_new, self.key_ts, self.key_radius[:, 1])
        ]).T

        intpl = self.interpolate(ts, radius=False)
        intpl['radius'] = rs1
        return intpl, ts_new


    def inverse_transform(self, samples_local, ts):
        x_proj = 2*ts - 1
        res = self.interpolate(ts)
        verts, yz_rs, frame = res['points'], res['radius'], res['frame']

        x_rs = self.calc_x_radius(ts)
        radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)
        # F(v - Pv) = (1,ry,rz)*(v_ - (v_x,0,0))
        # i.e. samples_local[:,0] - samples_x
        samples_global = samples_local.copy()
        samples_global[:, 0] -= x_proj
        samples_global *= radius
        # NOTE: here it should be the inverse of frames
        # so it is the transpose(since they are unitary mats)
        # and we simply modify it in einsum: nij,nj->ni => nji,nj->ni
        samples_global = np.einsum('nji,nj->ni', frame, samples_global)
        samples_global += verts
        return samples_global

    def generate_samples(self, num_samples):
        # mapping: cylinder <-> y^2+z^2 = 1, -1 <= x <= 1
        samples = np.random.uniform(-1., 1., size=(num_samples, 3))
        yz_norms = np.linalg.norm(samples[:,1:], axis=1)
        inside = yz_norms <= 1
        samples = samples[inside]

        # coords: [-1,1] to [0,1]
        samples_x = samples[:, 0]
        ts = (samples_x + 1) / 2
        res = self.interpolate(ts)
        verts, yz_rs, frame = res['points'], res['radius'], res['frame']

        # F(v - Pv) = (1,ry,rz)*(v_ - (v_x,0,0))
        # i.e. samples[:,0] - samples_x
        samples_global = np.zeros((samples.shape[0], 3))
        samples_global[:, 1:] = samples[:, 1:]*yz_rs
        # NOTE: here it should be the inverse of frames
        # so it is the transpose(since they are unitary mats)
        # and we simply modify it in einsum: nij,nj->ni => nji,nj->ni
        samples_global = np.einsum('nji,nj->ni', frame, samples_global)
        samples_global += verts

        # sdf_scales = np.sqrt(np.product(yz_rs, axis=1))
        return {
            'samples': samples_global,
            'samples_local':samples,
            'radius': yz_rs,
            'coords': ts,
        }


    def export_data(self):
        return {
            'key_points': self.key_points,
            'key_radius': self.key_radius,
            'z_axis': self.z_axis,
            'ball': {
                'start_x': self.start_ball_x,
                'end_x': self.end_ball_x,
            }
        }
    
    def export_vis(self):
        vidx = np.arange(self.key_points.shape[0])
        return {
            'vertices': self.key_points,
            'edges': np.asarray([vidx[:-1], vidx[1:]]).T,
        }
    
