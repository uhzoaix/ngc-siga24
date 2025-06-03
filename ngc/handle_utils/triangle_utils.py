import os, pickle
import numpy as np
import os.path as op
import trimesh
from time import time

class Triangle():
    """docstring for Triangle."""
    def __init__(self, arg=None):
        if arg is None:
            return
        # [v0, v1, v2]
        self.load(arg)

    def preprocess(self):
        self.p0 = self.points[0]
        self.p1 = self.points[1]
        self.p2 = self.points[2]
        self.e01 = self.p1 - self.p0
        self.e12 = self.p2 - self.p1
        self.e02 = self.p2 - self.p0
        self.edge_lengths = [
            np.linalg.norm(self.e01),
            np.linalg.norm(self.e12),
            np.linalg.norm(self.e02),
        ]

        self.vec01 = self.e01 / self.edge_lengths[0]
        self.vec12 = self.e12 / self.edge_lengths[1]
        self.vec02 = self.e02 / self.edge_lengths[2]

        # check if the triangle is valid
        self.ip_e1e2 = self.e01 @ self.e02
        if np.allclose(np.abs(self.vec01 @ self.vec02), 1.):
            raise ValueError('Not a Valid Triangle')

        self.normal = np.cross(self.e01, self.e02)
        self.normal /= np.linalg.norm(self.normal)

        self.edges = [[0,1], [1,2], [2,0]]
        self.edge_vecs = [self.vec01, self.vec12, -self.vec02]
        # local frame: p0 as origin, vec01 as x-axis, normal as z-axis
        self.origin = self.p0
        self.frame = np.asarray([
            self.vec01,
            np.cross(self.normal, self.vec01),
            self.normal
        ])
        e12_2d = self.frame @ self.vec12
        e20_2d = self.frame @ (-self.vec02)
        p1_2d = self.frame @ (self.p1 - self.origin)
        # p2_2d = self.frame @ (self.p2 - self.origin)
        b1 = e12_2d[1]*p1_2d[0] - e12_2d[0]*p1_2d[1]
        # b2 = e20_2d[1]*p2_2d[0] - e20_2d[0]*p2_2d[1]
        # assert np.allclose(b2, 0)
        
        # if line_func > 0, means point at the one side of triangle edge
        # points(N,3)(1 at 2dim), points @ lines2d, get the value
        self.lines2d = np.asarray([
            [0,1, 0],
            [-e12_2d[1], e12_2d[0], b1],
            [-e20_2d[1], e20_2d[0], 0],
        ]).T


    def get_centroid(self):
        return self.points.mean(axis=0)
    

    def update_triangle(self, points=None, radius=None):
        if points is not None:
            self.points = points
        if radius is not None:
            self.v_radius = radius
        
        self.preprocess()


    def centroid_coord(self, points_on_tri):
        # ps (N, 3)
        ps = points_on_tri - self.p0
        ip1 = ps @ self.e01
        ip2 = ps @ self.e02
        
        rhs = np.vstack([ip1, ip2])
        e01_sq = np.sum(self.e01**2)
        e02_sq = np.sum(self.e02**2)
        mat = np.asarray([
            [e02_sq, -self.ip_e1e2],
            [-self.ip_e1e2, e01_sq],
        ]) / (e01_sq*e02_sq - self.ip_e1e2**2)

        c1c2 = mat @ rhs
        c0 = 1 - np.sum(c1c2, axis=0)

        return np.concatenate([
            c0.reshape(1, -1),
            c1c2
        ], axis=0).T
    
    def check_centroid_coords(self, coords):
        # assume coords sum 1
        valid = np.logical_and(coords >= 0, coords <= 1)
        return np.all(valid, axis=1)
    

    def calc_bbox(self):
        pts = np.concatenate([
            self.points + self.v_radius[:, None],
            self.points - self.v_radius[:, None],
        ], axis=0)
        return np.max(pts, axis=0), np.min(pts, axis=0)
    

    def project_triangle(self, vs):
        ip = (vs - self.p0) @ self.normal
        pvs = vs - ip[:, None]* self.normal
        # ccoords: (N, 3)
        coords = self.centroid_coord(pvs)
        rs = coords @ self.v_radius
        ds = np.linalg.norm(vs - pvs, axis=1)
        return {
            'sdf': ds - rs,
            'proj': pvs,
            'coord': coords,
            'radius': rs,
        }

    def project_edge(self, eid, vs, ip):
        n_vs = vs.shape[0]
        vid0, vid1 = self.edges[eid]
        p0, p1 = self.points[vid0], self.points[vid1]
        r0, r1 = self.v_radius[vid0], self.v_radius[vid1]

        pvs = p0 + ip[:, None]*(p1 - p0)
        ds = np.linalg.norm(vs-pvs, axis=1)
        rs = r0 + ip*(r1 - r0)
        coords = np.zeros((n_vs, 3))
        coords[:, vid0] = 1 - ip
        coords[:, vid1] = ip
        return {
            'sdf': ds - rs,
            'proj': pvs,
            'coord': coords,
            'radius': rs,
        }

    def project_vertex(self, vid, vs):
        n_vs = vs.shape[0]
        ds = np.linalg.norm(vs - self.points[vid], axis=1)
        rs = self.v_radius[vid]* np.ones(n_vs)
        pvs = np.tile(self.points[vid], (n_vs, 1))
        coords = np.zeros((n_vs, 3))
        coords[:, vid] = 1
        return {
            'sdf': ds - rs,
            'proj': pvs,
            'coord': coords,
            'radius': rs,
        }
    
    def line_projection(self, eid, vs):
        vid = eid
        p0 = self.points[vid]
        vec = self.edge_vecs[eid]
        edge_len = self.edge_lengths[eid]

        ip = (vs - p0) @ vec
        ip /= edge_len
        return ip
    
    def __nearest_res_update(self, final_res, res, idx):
        final_res['sdf'][idx] = res['sdf']
        final_res['proj'][idx] = res['proj']
        final_res['coord'][idx] = res['coord']
        final_res['radius'][idx] = res['radius']

    def calc_nearest(self, vs, only_return_sdf=False):
        n_vs = vs.shape[0]
        final_res = {
            'sdf': np.zeros(n_vs),
            'proj': np.zeros((n_vs, 3)),
            'coord': np.zeros((n_vs, 3)),
            'radius': np.zeros(n_vs)
        }
        vs2d_h = (vs - self.origin) @ (self.frame.T)
        # vs2d_h(n_vs,3)
        vs2d_h[:, 2] = 1

        line_val = vs2d_h @ self.lines2d
        area_code = line_val >= 0
        # [T,T,T]: on triangle; F: at outside of triangle edge

        marks = np.ones(n_vs, dtype=bool)
        # print('Start, marks remain: ', np.sum(marks))
        vidx = np.arange(n_vs)
        tri_face = np.all(area_code, axis=1)
        vidx_face = vidx[tri_face]
        res = self.project_triangle(vs[vidx_face])
        self.__nearest_res_update(final_res, res, vidx_face)
        marks[vidx_face] = False
        # print('Face processed')
        # print('marks remain: ', np.sum(marks))

        # for convenience, reverse T/F
        area_code = np.logical_not(area_code)
        edge_val = np.zeros((n_vs, 3))
        for eid in range(3):
            vidx_side = vidx[area_code[:, eid]]
            # value of projected point parameters
            vals = self.line_projection(eid, vs[vidx_side])
            edge_val[vidx_side, eid] = vals
            inside = np.logical_and(vals >= 0, vals <= 1)
            # project to edge
            vidx_edge = vidx_side[inside]
            vals_edge = vals[inside]
            res = self.project_edge(eid, vs[vidx_edge], vals_edge)
            self.__nearest_res_update(final_res, res, vidx_edge)
            marks[vidx_edge] = False
            # print(f'Edge{eid} processed')
            # print('marks remain: ', np.sum(marks))

        # find points nearest to three vertices
        vidx_res = vidx[marks]
        egval_res = edge_val[marks]
        for vid in range(3):
            eid1, eid0 = vid-1, vid
            inside = np.logical_or(egval_res[:, eid1]>1, egval_res[:, eid0]<0)
            vid_idx = vidx_res[inside]

            res = self.project_vertex(vid, vs[vid_idx])
            self.__nearest_res_update(final_res, res, vid_idx)
            marks[vid_idx] = False
            # print(f'Vertex{vid} processed')
            # print('marks remain: ', np.sum(marks))

        if np.any(marks):
            raise ValueError('Still some points not processed')
        
        if only_return_sdf:
            return final_res['sdf']
        else:
            return final_res
        
        
    def calc_frame(self, projs, coords, mode='circle'):
        n_v = projs.shape[0]
        centroid = self.get_centroid()
        if mode == 'circle':
            vec_to_cent = centroid - projs
            dists = np.linalg.norm(vec_to_cent, axis=1)
            zero_flag = np.allclose(dists, 0)
            
            if np.any(zero_flag):
                vidx = np.arange(n_v)
                zero_idx = vidx[zero_flag]
                centroid_yaxis = centroid - self.points[0]
                vec_to_cent[zero_idx] = np.tile(centroid_yaxis, (zero_idx.shape[0], 1))
                dists[zero_idx] = np.linalg.norm(centroid_yaxis)

            y_axis = vec_to_cent / dists[:, None]
            z_axis = np.tile(self.normal, (n_v, 1))
            x_axis = np.cross(y_axis, z_axis)
            frames = np.concatenate([
                x_axis.reshape(n_v, 1, 3),
                y_axis.reshape(n_v, 1, 3),
                z_axis.reshape(n_v, 1, 3),
            ], axis=1)
            return frames, None
        
        if mode == 'spiral':
            codes = np.zeros(n_v, dtype=int)
            avg = 1./3
            for i in range(n_v):
                c0, c1, c2 = coords[i]
                if c0 < avg and c0 < c1 < 1-2*c0 and c0 < c2 < 1-2*c0:
                    codes[i] = 0
                elif c1 < avg and c1 < c0 < 1-2*c1 and c1 < c2 < 1-2*c1:
                    codes[i] = 1
                elif c2 < avg and c2 < c1 < 1-2*c2 and c2 < c0 < 1-2*c2:
                    codes[i] = 2
                elif c0 > avg and c1 == c2:
                    codes[i] = 3
                elif c1 > avg and c0 == c2:
                    codes[i] = 4
                elif c2 > avg and c0 == c1:
                    codes[i] = 5
                else:
                    # centroid case: c0=c1=c2
                    codes[i] = 3
            
            cv = self.points - centroid[None, :]
            cv /= np.linalg.norm(cv, axis=1, keepdims=True)
            code_vecs = [
                self.edge_vecs[1], self.edge_vecs[2], self.edge_vecs[0],
                cv[0], cv[1], cv[2]
            ]
            # code_vecs(6, 3)
            code_vecs = np.vstack(code_vecs)
            # x_axis (nv, 3)
            x_axis = code_vecs[codes]
            z_axis = np.tile(self.normal, (n_v, 1))
            y_axis = np.cross(z_axis, x_axis)
            frames = np.concatenate([
                x_axis.reshape(n_v, 1, 3),
                y_axis.reshape(n_v, 1, 3),
                z_axis.reshape(n_v, 1, 3),
            ], axis=1)
            return frames, codes
        
    
    def localize_samples(self, samples):
        nearest_res = self.calc_nearest(samples)
        sdf = nearest_res['sdf']
        inside = sdf <= 0

        projs = nearest_res['proj'][inside]
        coords = nearest_res['coord'][inside]
        rs = nearest_res['radius'][inside]
        frames, codes = self.calc_frame(projs, coords, mode='spiral')

        samples = samples[inside]
        diff = (samples - projs) / rs[:, None]
        samples_local = np.einsum('nij,nj->ni', frames, diff)
        return {
            'samples': samples,
            'samples_local': samples_local,
            'projs': projs,
            'coords': coords,
            'rs': rs,
            'frames': frames,
            'codes': codes,
        }, inside
    

    def get_disc_points(self, n_pts):
        arc = 2*np.pi / n_pts
        thetas = arc*np.arange(n_pts)
        pts2d_x = np.cos(thetas)
        pts2d_y = np.sin(thetas)

        x_axis = self.edge_vecs[0]
        y_axis = np.cross(self.normal, x_axis)

        pts = x_axis*pts2d_x[:,None] + y_axis*pts2d_y[:, None]
        pts = np.concatenate([
            np.zeros((1,3)), 
            pts,
            pts / 2
        ], axis=0)
        return pts
    
    def specify_disc_size(self, samples, projs, rs, disc_size, thres):
        def inv_dist(arr):
            inv = np.zeros(arr.shape)
            non_zero = arr > 0
            inv[non_zero] = 1. / arr[non_zero]
            return inv

        def calc_keypoints(_ps, _pbars, _rs, _r):
            # intersected points of disc and line to three vertices
            # print(f'ps:{_ps.shape}, pbars:{_pbars.shape}, rs:{_rs.shape}, r:{_r}')
            # vecs:(Ns, 3, 3)
            vecs = self.points[None, :, :] - _pbars[:, None, :]
            # print(f'vecs: {vecs.shape}')
            ds = np.linalg.norm(vecs, axis=-1)
            inv_ds = _r* inv_dist(ds)
            # print(f'ds: {ds.shape}')
            keypoints = _pbars[:, None, :] + vecs* inv_ds[...,None]
            # print(f'keypoints: {keypoints.shape}')
            key_rs = _rs[:,None] + (self.v_radius[None, :] - _rs[:, None])* inv_ds
            # print(f'key_rs: {key_rs.shape}')

            # calculate farest point for all ps
            ip = (_ps - _pbars) @ self.normal
            _ps_tri = _ps - ip[:, None]* self.normal
            vec = _pbars - _ps_tri
            vec_norm = np.linalg.norm(vec, axis=1)
            inv_norm = _r* inv_dist(vec_norm)
            farest_ps = _pbars + vec*inv_norm[:, None]
            # radius of farest point, set large value for outside triangle points
            coords = self.centroid_coord(farest_ps)
            valid = self.check_centroid_coords(coords)
            invalid = np.logical_not(valid)
            farest_rs = coords @ self.v_radius
            farest_rs[invalid] = 10*(self.v_radius.max())

            keypoints = np.concatenate([keypoints, farest_ps[:,None,:]], axis=1)
            key_rs = np.concatenate([key_rs, farest_rs[:,None]], axis=1)
            return keypoints, key_rs
        
        ns = samples.shape[0]
        sidx = np.arange(samples.shape[0])
        current_disc_size = disc_size
        samples_disc_size = np.ones(ns)
        while len(sidx) > 0:
            temp_ps = samples[sidx]
            temp_pbar = projs[sidx]
            temp_rs = rs[sidx]
            # print(f'temp ps:{temp_ps.shape}, pbar:{temp_pbar.shape}, rs:{temp_rs.shape}')
            temp_keypts, temp_keyrs = calc_keypoints(
                temp_ps, temp_pbar, temp_rs, current_disc_size)
            
            # print(f'key pts:{temp_keypts.shape}, rs:{temp_keyrs.shape}')
            ds = np.linalg.norm(temp_ps[:,None,:] - temp_keypts, axis=-1)
            invalid = np.any(ds > temp_keyrs, axis=1)
            valid = np.logical_not(invalid)
            sidx_valid = sidx[valid] 
            samples_disc_size[sidx_valid] = current_disc_size
            if np.sum(invalid) == 0:
                break
            elif current_disc_size < thres:
                sidx_invalid = sidx[invalid]
                samples_disc_size[sidx_invalid] = current_disc_size
                break
            else:
                sidx = sidx[invalid]
                current_disc_size = current_disc_size / 2
        
        return samples_disc_size

    
    def localize_samples_disc(self, samples):
        nearest_res = self.calc_nearest(samples)
        sdf = nearest_res['sdf']
        inside = sdf <= 0

        samples = samples[inside]
        samples_projs = nearest_res['proj'][inside]
        samples_rs = nearest_res['radius'][inside]
        disc_pts = self.get_disc_points(n_pts=8)
        n_s = samples.shape[0]
        n_d = disc_pts.shape[0]
        # init_disc_size = np.mean(samples_rs + sdf[inside]) / 2
        init_disc_size = np.mean(self.v_radius) / 4
        # since pytorch cuda float precision is 1e-6
        disc_size_thres = 1e-5

        # specify disc_size for all samples
        samples_disc_size = self.specify_disc_size(
            samples, samples_projs, samples_rs, init_disc_size, disc_size_thres)

        print('disc size: init:{:.4f}, max:{:.4f}, min:{:.8f}, mean:{:.4f}'.format(
            init_disc_size, samples_disc_size.max(),
            samples_disc_size.min(), samples_disc_size.mean()
        ))
        disc_pts = samples_disc_size[:, None, None]* disc_pts[None,:,:]
        disc_samples = samples_projs[:, None, :] + disc_pts
        disc_samples = disc_samples.reshape(n_s*n_d, 3)
        nearest_res = self.calc_nearest(disc_samples)

        # for disc samples
        projs = nearest_res['proj']
        coords = nearest_res['coord']
        rs = nearest_res['radius']
        frames, codes = self.calc_frame(projs, coords, mode='spiral')

        samples_tile = np.tile(samples[:, None, :], (1, n_d, 1))
        samples_tile = samples_tile.reshape(n_s*n_d, 3)
        # make sure samples in ball of projs
        ds = np.linalg.norm(samples_tile - projs, axis=1)
        outlier = ds > rs
        if np.any(outlier):
            print('outlier num: ', np.sum(outlier))
            diff = ds[outlier] - rs[outlier]
            print('dist diff: mean:{:.6f}, max:{:.6f}, min:{:.6f}'.format(
                diff.mean(), diff.max(), diff.min()
            ))
            # raise ValueError('Too large disc size')
        
        diff = (samples_tile - projs) / rs[:, None]
        samples_local = np.einsum('nij,nj->ni', frames, diff)
        return {
            'samples': samples,
            'samples_local': samples_local.reshape(n_s, n_d, 3),
            'projs': projs.reshape(n_s, n_d, 3),
            'coords': coords.reshape(n_s, n_d, 3),
            'rs': rs.reshape(n_s, n_d),
            'frames': frames.reshape(n_s, n_d, 3, 3),
            'codes': codes.reshape(n_s, n_d),
        }, inside
    

    def SDF_sampling(self, arg):
        shape_path = arg['shape_path']
        num_samples = arg['num_samples']

        shape = trimesh.load(shape_path, process=False)
        proxy_shape = trimesh.proximity.ProximityQuery(shape)

        bmax, bmin = self.calc_bbox()
        samples = np.random.uniform(0, 1, size=(num_samples, 3))
        samples = samples* (bmax - bmin) + bmin
        if arg['disc_samples']:
            data, _ = self.localize_samples_disc(samples)
        else:
            data, _ = self.localize_samples(samples)

        sdf_shape = -1*proxy_shape.signed_distance(data['samples'])
        data['sdf'] = sdf_shape
        return data
    
    def filter_grid(self, mc_grid, disc_samples=False):
        bmax, bmin = self.calc_bbox()
        samples, kidx = mc_grid.generate_samples_bbox(bmin, bmax)
        if disc_samples:
            data, inside = self.localize_samples_disc(samples)
        else:
            data, inside = self.localize_samples(samples)
        kidx = kidx[inside]

        return data, kidx

        
    def to_std_coords(self, std_tri, vs, projs, coords, rs):
        n_v = vs.shape[0]
        centroid = self.get_centroid()
        # standard Triangle
        std_verts = std_tri.points
        std_rs = std_tri.v_radius
        std_normal = std_tri.normal
        std_centroid = std_tri.get_centroid()

        # # check coords
        # check_c = np.logical_and(coords >= 0, coords <= 1)
        # check_sum = np.sum(coords, axis=1) == 1
        # print(f'coords valid: {np.all(check_c)}, sum:{np.all(check_sum)}')

        Pvs_std = coords @ std_verts
        rs_std = coords @ std_rs

        scales = rs_std / rs
        
        # calculate y axis for all vs
        vec_to_cent = centroid - projs
        dists = np.linalg.norm(vec_to_cent, axis=1)
        zero_flag = np.allclose(dists, 0)
        
        if np.any(zero_flag):
            vidx = np.arange(n_v)
            zero_idx = vidx[zero_flag]
            centroid_yaxis = centroid - self.points[0]
            vec_to_cent[zero_idx] = np.tile(centroid_yaxis, (zero_idx.shape[0], 1))
            dists[zero_idx] = np.linalg.norm(centroid_yaxis)

        y_axis = vec_to_cent / dists[:, None]
        z_axis = np.tile(self.normal, (n_v, 1))
        x_axis = np.cross(y_axis, z_axis)
        frame = np.concatenate([
            x_axis.reshape(n_v, 1, 3),
            y_axis.reshape(n_v, 1, 3),
            z_axis.reshape(n_v, 1, 3),
        ], axis=1)

        # calculate frames in std triangle
        std_vec2cent = std_centroid - Pvs_std
        std_dists = np.linalg.norm(std_vec2cent, axis=1)
        
        if np.any(zero_flag):
            # zero_idx should be same
            std_cyaxis = std_centroid - std_tri.points[0]
            std_vec2cent[zero_idx] = np.tile(std_cyaxis, (zero_idx.shape[0], 1))
            std_dists[zero_idx] = np.linalg.norm(std_cyaxis)

        std_yaxis = std_vec2cent / std_dists[:, None]
        std_zaxis = np.tile(std_normal, (n_v, 1))
        std_xaxis = np.cross(std_yaxis, std_zaxis)
        std_frame = np.concatenate([
            std_xaxis.reshape(n_v, 1, 3),
            std_yaxis.reshape(n_v, 1, 3),
            std_zaxis.reshape(n_v, 1, 3),
        ], axis=1)

        pidx = np.arange(n_v)
        # check = []
        # for i in range(n_v):
        #     ortho_check = np.allclose(frame[i] @ frame[i].T, np.eye(3))
        #     check.append(ortho_check)
        # print(f'check: {np.all(check)}, num of failed:{n_v - np.sum(check)}')

        # check = []
        # for i in range(n_v):
        #     ortho_check = np.allclose(std_frame[i] @ std_frame[i].T, np.eye(3))
        #     check.append(ortho_check)
        # print(f'check std: {np.all(check)}, num of failed:{n_v - np.sum(check)}')

        # exact the pointwise rigid+scale transformation
        diff = vs - projs
        # dist = np.linalg.norm(diff, axis=1)
        local_coords = np.einsum('nij,nj->ni', frame, diff)
        local_coords *= scales[:, None]
        # NOTE: nij,ni->nj: acutall the transpose of matrix 
        # that is, to apply the inverse of trans matrix std_frame
        vs_std = np.einsum('nij,ni->nj', std_frame, local_coords)
        # dist_std = np.linalg.norm(vs_std, axis=1)
        vs_std += Pvs_std
        return vs_std, scales, Pvs_std
    

    def load(self, arg):
        self.points = arg['points']
        self.v_radius = arg['radius']
        self.preprocess()

    def export_data(self):
        return {
            'points': self.points,
            'radius': self.v_radius,
        }

def test_stri_mesh(output_path):
    from .mc_utils import MCGrid
    name = 'pose2'
    scale = 1
    r0, r1, r2 = [0.3, 0.3, 0.3]
    jp = np.asarray([0.5,0.,0.])
    cp1 = np.asarray([0., 0.5, 0.])
    cp2 = np.asarray([0., -0.5, 0.5])
    tri_pts = np.asarray([jp, cp1, cp2])
    radius = np.asarray([r0, r1, r2])
    tri = Triangle({
        'points': tri_pts / scale,
        'radius': radius / scale,
    })

    level = 0.
    mc_config = {
        'reso': 64,
        'level': level
    }
    mc_grid = MCGrid(mc_config)
    samples, samples_kid = mc_grid.generate_samples()
    vals = tri.calc_nearest(samples, only_return_sdf=True)
    mc_grid.update_grid(vals, samples_kid)
    mesh = mc_grid.extract_mesh()
    mesh_path = op.join(output_path, f'{name}_L{level}.ply')
    mesh.export(mesh_path)

    tri_curve = {
        'vertices': tri.points,
        'edges': tri.edges,
    }
    with open(op.join(output_path, f'{name}_tri_curve.pkl'), 'wb') as f:
        pickle.dump(tri_curve, f)

    print('Done')


def make_projection_pair(vs, projs):
    n_v = vs.shape[0]
    verts = np.concatenate([vs, projs], axis=0)
    edges = np.asarray([np.arange(n_v), np.arange(n_v, 2*n_v)]).T
    return {
        'vertices': verts,
        'edges': edges
    }

def test_std_coords(output_path):
    std_verts = np.asarray([
        [0.5, 0,0],
        [-0.5, 0.5,0],
        [-0.5, -0.5,0],
    ])
    std_radius = np.asarray([0.5, 0.5, 0.5])
    std_tri = Triangle({
        'points': std_verts,
        'radius': std_radius,
    })

    scale = 2
    r0, r1, r2 = [0.3, 0.3, 0.3]
    jp = np.asarray([0.,0.,0.])
    cp1 = np.asarray([0., 1., 0.])
    cp2 = np.asarray([-1., -1, 0.])
    tri_pts = np.asarray([jp, cp1, cp2])
    radius = np.asarray([r0, r1, r2])
    tri = Triangle({
        'points': tri_pts / scale,
        'radius': radius / scale,
    })
    mesh_name = 'tri1_L0.0.ply'
    mesh = trimesh.load(op.join(output_path, mesh_name), process=False)
    V = np.asarray(mesh.vertices)

    res = tri.calc_nearest(V)

    projs = res['proj']
    coords = res['coord']
    rs = res['radius']

    coords_div = coords > 0
    e2_c = np.asarray([1, 0, 1], dtype=bool)
    e2_c = np.tile(e2_c, (V.shape[0], 1))
    print(coords_div.shape, coords_div[0])
    print(e2_c.shape, e2_c[0])
    flag = np.all(coords_div == e2_c, axis=1)

    V_std,_,pv_std = tri.to_std_coords(std_tri, V[flag], projs[flag], coords[flag], rs[flag])
    # mesh_std = trimesh.Trimesh(V_std, process=False)
    # mesh_std.export(op.join(output_path, 'tri1_e2_std.ply'))

    # mesh_v = trimesh.Trimesh(V[flag], process=False)
    # mesh_v.export(op.join(output_path, 'tri1_e2_v.ply'))
    vis_v = make_projection_pair(V[flag], projs[flag])
    vis_std = make_projection_pair(V_std, pv_std)
    with open(op.join(output_path, 'tri1_e2_vproj.pkl'), 'wb') as f:
        pickle.dump(vis_v, f)
        
    with open(op.join(output_path, 'tri1_e2_stdproj.pkl'), 'wb') as f:
        pickle.dump(vis_std, f)


def process_data():
    shape_path = '/data/triangle_test/cheese.ply'
    num_samples = 2000
    output_path = '/data/triangle_test'
    os.makedirs(output_path, exist_ok=True)

    t0 = time()
    arg = {
        'shape_path': shape_path,
        'num_samples': num_samples,
        'disc_samples': True,
    }
    std_verts = np.asarray([
        [0.5, 0,0],
        [-0.5, 0.5,0],
        [-0.5, -0.5,0],
    ])
    std_radius = np.asarray([0.5, 0.5, 0.5])
    std_tri = Triangle({
        'points': std_verts,
        'radius': std_radius,
    })
    # tri_curve = {
    #     'vertices': std_tri.points,
    #     'edges': std_tri.edges,
    # }
    # file_name = f'std_tri_curve.pkl'
    # out_file = f'/home/uhzoaix/Project/control_exp/generation/triangle/{file_name}'
    # with open(out_file, 'wb') as f:
    #     pickle.dump(tri_curve, f)

    # tri_data = std_tri.export_data()
    # with open(op.join(output_path, 'triangle.pkl'), 'wb') as f:
    #     pickle.dump(tri_data, f)

    data = std_tri.SDF_sampling(arg)
    with open(op.join(output_path, 'sdf_data_disc2.pkl'), 'wb') as f:
        pickle.dump(data, f)

    print('Done, time cost: ', time()-t0)

def debug_disc_sampling():
    std_verts = np.asarray([
        [0.5, 0,0],
        [-0.5, 0.5,0],
        [-0.5, -0.5,0],
    ])
    std_radius = np.asarray([0.5, 0.5, 0.5])
    std_tri = Triangle({
        'points': std_verts,
        'radius': std_radius,
    })

    samples = np.asarray([
        [0,0,0.3],
        [0,0,-0.1],
        [0,0.5,0.2],
        [0,-0.5,0.2],
        [-0.6,0,0.2],
        [0.6,0,0.2],
        [0.6,0.1,0.2],
        [0.6,-0.1,0.2],
    ])
    data, _ = std_tri.localize_samples_disc(samples)
    data['triangle'] = {
        'vertices': std_tri.points,
        'edges': std_tri.edges,
    }
    samples_local = data['samples_local']
    samples_rs = data['rs']
    for i in range(samples.shape[0]):
        print('local pts: ', samples_local[i])
        print('radius: ', samples_rs[i])
        print('-------------------------------')

    raise ValueError

    root_path = '/home/uhzoaix/Project/control_exp/generation/triangle'
    output_path = op.join(root_path, 'debug')
    os.makedirs(output_path, exist_ok=True)
    with open(op.join(output_path, 'disc_sampling.pkl'), 'wb') as f:
        pickle.dump(data, f)

    print('Done')

if __name__ == "__main__":
    root_path = '/home/uhzoaix/Project/control_exp/generation/triangle'
    output_path  = root_path
    os.makedirs(output_path, exist_ok=True)

    # test_stri_mesh(output_path)
    process_data()
    # debug_disc_sampling()