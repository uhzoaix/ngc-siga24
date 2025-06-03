import os, pickle
import numpy as np
import trimesh
import skimage
import os.path as op


class MCGrid():
    """docstring for MCGrid."""
    def __init__(self, arg):
        N = arg['reso']
        self.reso = N
        N1 = N + 1
        origin = np.asarray([-1., -1., -1.])
        step = 2. / N
        k_basis = np.asarray([1, N1, N1**2])
        self.grid_config = {
            'level': arg['level'],
            'origin': origin,
            'reso': N,
            'step': step,
            'k_basis': k_basis,
            'do_flip': True,
        }

        # 10: large positive number for background vals
        self.val_grid = 10* np.ones(N1**3)
        self.empty_marks = np.ones(N1**3, dtype=bool)


    def update_grid(self, val, kidx, mode='overwrite', mark=False):
        if mode == 'minimum':
            # TODO: inplace minimum
            self.val_grid[kidx] = np.minimum(self.val_grid[kidx], val)
        elif mode == 'overwrite':
            self.val_grid[kidx] = val
        elif mode == 'min_nonmark':
            kidx_mark = self.empty_marks[kidx]
            kidx_empty = kidx[kidx_mark]
            val_empty = val[kidx_mark]
            self.val_grid[kidx_empty] = np.minimum(
                self.val_grid[kidx_empty], val_empty)
        else:
            raise NotImplementedError
        
        if mark:
            self.empty_marks[kidx] = False
    
    def update_grid_func(self, val, kidx, func=None):
        if not hasattr(self, 'func_marks'):
            self.func_marks = self.create_marks()
        
        marks = self.func_marks[kidx]
        marks_false = np.logical_not(marks)
        kidx_true = kidx[marks]
        kidx_false = kidx[marks_false]

        self.val_grid[kidx_false] = val[marks_false]

        if np.any(marks):
            vals_true = self.val_grid[kidx_true]
            # diff_vals = vals_true - val[marks]
            if func is None:
                self.val_grid[kidx_true] = val[marks]
            else:
                self.val_grid[kidx_true] = func(val[marks], vals_true)

        self.func_marks[kidx] = True
        # return diff_vals, kidx_true
    
    def clear_grid(self, val=10.):
        self.val_grid[:] = val
        self.empty_marks[:] = True
        if hasattr(self, 'func_marks'):
            self.func_marks[:] = False

    @staticmethod
    def get_plane(arg):
        N = arg['reso']
        N1 = N + 1
        origin = np.asarray([-1., -1.])
        step = 2. / N
        k_basis = np.asarray([1, N1])

        kidx = np.arange(N1**2)
        samples_j = kidx // k_basis[1]
        samples_i = kidx % k_basis[1]
        samples_ij = np.stack([
            samples_i, samples_j
        ]).T

        samples2d = samples_ij*step + origin
        samples = np.zeros((samples2d.shape[0], 3))
        # samples[:,:2] = samples2d
        # samples[:, 2] = 0.12
        # samples[:, 1:] = samples2d
        # samples[:, 0] = -0.2
        samples[:, 0] = samples2d[:,0]
        samples[:, 2] = samples2d[:,1]
        return samples
    
    def get_config(self):
        return self.grid_config
    
    def get_marked_intersection(self, kidx):
        marks = self.empty_marks[kidx]
        intsct = np.logical_not(marks)
        kidx_int = kidx[intsct]
        vals = self.val_grid[kidx_int]
        return vals, kidx_int, intsct
    
    def get_vals(self, kidx):
        return self.val_grid[kidx]
    
    def generate_samples(self, min_corner=None, max_corner=None, only_kid=False):
        N = self.reso
        step = self.grid_config['step']
        origin = self.grid_config['origin']
        k_basis = self.grid_config['k_basis']
        if max_corner is None:
            max_corner = np.asarray([N, N, N])
        else:
            max_corner = np.clip(max_corner, a_min=None, a_max=N).astype(int)
        
        if min_corner is None:
            min_corner = np.asarray([0, 0, 0])
        else:
            min_corner = np.clip(min_corner, a_min=0, a_max=None).astype(int)

        xr = np.arange(min_corner[0], max_corner[0]+1)
        yr = np.arange(min_corner[1], max_corner[1]+1)
        zr = np.arange(min_corner[2], max_corner[2]+1)
        X, Y, Z = np.meshgrid(xr, yr, zr)
        sample_ijk = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
        samples_kid = (sample_ijk.astype(k_basis.dtype)) @ k_basis
        if only_kid:
            return samples_kid
        
        samples = sample_ijk*step + origin
        return samples, samples_kid

    def generate_samples_bbox(self, bmin, bmax):
        origin = self.grid_config['origin']
        step = self.grid_config['step']
        max_corner = np.floor((bmax - origin) / step) + 1
        min_corner = np.floor((bmin - origin) / step)
        return self.generate_samples(min_corner, max_corner)
    

    def filter_grid_ball(self, p, r):
        bmin = p - r
        bmax = p + r
        samples, kidx = self.generate_samples_bbox(bmin, bmax)
        ds = np.linalg.norm(samples-p, axis=1)
        inside = ds <= r
        return samples[inside], kidx[inside]

    def create_marks(self):
        N1 = self.reso + 1
        return np.zeros(N1**3, dtype=bool)
    

    def mark_bbox(self, bmax, bmin, marks=None, extend_bbox=False):
        origin = self.grid_config['origin']
        step = self.grid_config['step']
        max_corner = np.floor((bmax - origin) / step) + 1
        min_corner = np.floor((bmin - origin) / step)
        
        if extend_bbox:
            max_corner += 1
            min_corner -= 1

        kidx = self.generate_samples(min_corner, max_corner, only_kid=True)
        if marks is None:
            self.empty_marks[kidx] = False
        else:
            marks[kidx] = True

    def get_marked(self, marks=None):
        origin = self.grid_config['origin']
        step = self.grid_config['step']
        k_basis = self.grid_config['k_basis']

        if marks is None:
            marks = np.logical_not(self.empty_marks)

        kidx = np.argwhere(marks).flatten()
        samples_k = kidx // k_basis[2]
        samples_j = (kidx - k_basis[2]*samples_k) // k_basis[1]
        samples_i = kidx % k_basis[1]
        samples_ijk = np.stack([
            samples_i, samples_j, samples_k
        ]).T

        samples = samples_ijk*step + origin
        return samples, kidx
    
    def idx2pts(self, kidx):
        origin = self.grid_config['origin']
        step = self.grid_config['step']
        k_basis = self.grid_config['k_basis']

        samples_k = kidx // k_basis[2]
        samples_j = (kidx - k_basis[2]*samples_k) // k_basis[1]
        samples_i = kidx % k_basis[1]
        samples_ijk = np.stack([
            samples_i, samples_j, samples_k
        ]).T

        samples = samples_ijk*step + origin
        return samples

    def voxelize_points(self, points):
        origin = self.grid_config['origin']
        step = self.grid_config['step']
        k_basis = self.grid_config['k_basis']
        
        ijk = ((points - origin) / step).astype(int)
        kid = ijk @ k_basis

        N1 = self.reso + 1
        voxel = np.zeros(N1**3, dtype=bool)
        voxel[kid] = True
        return voxel

    def extract_mesh(self, face_flip=True):
        config = self.grid_config
        N1 = self.reso + 1
        sdf_val = self.val_grid.reshape(N1, N1, N1)
        verts, faces, _, _ = skimage.measure.marching_cubes(
            sdf_val,
            level=config['level'],
            spacing=[config['step']] * 3
        )

        verts = verts + config['origin']
        # move the origin and flip x-z axis
        if config['do_flip']:
            verts_align = np.zeros_like(verts)
            verts_align[:, 0] = verts[:, 2]
            verts_align[:, 1] = verts[:, 1]
            verts_align[:, 2] = verts[:, 0]

        if face_flip:
            faces = faces[:, [0, 2, 1]]

        mesh = trimesh.Trimesh(
            vertices=verts_align,
            faces=faces,
            process=False
        )
        return mesh

# Temp  
# def calc_sdf_batch(self, proxy_shape, samples, batch_size):
#     n_samples = samples.shape[0]
#     sdf_vals = np.zeros(n_samples)

#     start = 0
#     n_batch = int(np.ceil(n_samples / batch_size))
#     for i in range(n_batch):
#         end = min(n_samples, start + batch_size)
#         vals = proxy_shape.signed_distance(samples[start: end])
#         sdf_vals[start: end] = -1*vals
    
#         start += batch_size

#     return sdf_vals
