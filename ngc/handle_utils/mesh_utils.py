import os, pickle
import trimesh
import numpy as np
import os.path as op


class CylindersMesh():
    """docstring for CylindersMesh."""
    def __init__(self):
        self.vertices = []
        self.faces = []

    def add_cylinder(self, cylinder_samples):
        # cylinder_samples: (N_t, N_theta, 3)
        num_vert = sum([verts.shape[0] for verts in self.vertices])
        n_t, n_theta = cylinder_samples.shape[0], cylinder_samples.shape[1]
        samples = cylinder_samples.reshape(-1, 3)
        self.vertices.append(cylinder_samples.reshape(-1, 3))
        self.cylinder_samples = cylinder_samples
        num_samples = samples.shape[0]

        vidx = np.arange(num_vert, num_vert+num_samples)
        # add faces
        vidx_cyl = vidx.reshape(n_t, n_theta)
        vidx_cyl_next = np.roll(vidx_cyl, shift=-1, axis=1)
        fidx1 = np.vstack([
            vidx_cyl[:-1], 
            vidx_cyl_next[:-1],
            vidx_cyl_next[1:],
        ]).reshape(3, -1).T

        fidx2 = np.vstack([
            vidx_cyl_next[1:],
            vidx_cyl[1:], 
            vidx_cyl[:-1],
        ]).reshape(3, -1).T
        
        self.faces.append(fidx1)
        self.faces.append(fidx2)
        return vidx_cyl
    

    def add_cap(self, center, circle_vidx, flip_face=False):
        center_vid = sum([verts.shape[0] for verts in self.vertices])
        self.vertices.append(center.reshape(1, 3))
        circle_vidx_next = np.roll(circle_vidx, shift=1)
        cvid = np.repeat(center_vid, circle_vidx.shape)
        cap_faces = np.vstack([
            circle_vidx,
            circle_vidx_next,
            cvid,
        ]).T
        if flip_face:
            cap_faces = cap_faces[:, [0, 2, 1]]
        self.faces.append(cap_faces)

    def filter_grid(self, mc_grid, extend_bbox=False):
        n_curve = self.cylinder_samples.shape[0]

        marks = mc_grid.create_marks()
        for i in range(n_curve-1):
            part = self.cylinder_samples[i:i+2].reshape(-1, 3)
            bmax, bmin = part.max(axis=0), part.min(axis=0)

            mc_grid.mark_bbox(bmax, bmin, marks, extend_bbox)

        samples, kidx = mc_grid.get_marked(marks)
        return samples, kidx

    def calc_bbox(self):
        verts = self.cylinder_samples.reshape(-1, 3)
        return verts.max(axis=0), verts.min(axis=0)

    def extract_mesh(self):
        self.vertices = np.concatenate(self.vertices, axis=0)
        self.faces = np.concatenate(self.faces, axis=0)
        mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)
        return mesh


if __name__ == "__main__":
    pass