import os,sys, pickle
import os.path as op
current_path = op.dirname(op.abspath(__file__))
sys.path.append(op.dirname(current_path))
import numpy as np
from blend_func import SmoothMaxMin

class Cylinder():
    """docstring for Cylinder."""
    def __init__(self, arg):
        self.p0 = arg['p0']
        self.p1 = arg['p1']
        self.r = arg['radius']
        self.ord = arg['ord']
        
        self.vec = self.p1 - self.p0
        self.length = np.linalg.norm(self.vec)
        self.vec /= self.length

    def project_line(self, points):
        projs = (points - self.p0) @ self.vec
        ts = projs / self.length
        ts = np.clip(ts, a_min=0., a_max=1.)
        return ts
    
    def interpolate(self, ts):
        return np.outer(ts, self.p1) + np.outer(1-ts, self.p0)
    
    def calc_bbox(self):
        points = [
            self.p0 + self.r,
            self.p0 - self.r,
            self.p1 + self.r,
            self.p1 - self.r,
        ]
        points = np.asarray(points)
        bmin, bmax = np.min(points,axis=0), np.max(points,axis=0)
        return bmin, bmax
    
    def filter_grid(self, mc_grid):
        bmin, bmax = self.calc_bbox()
        return mc_grid.generate_samples_bbox(bmin, bmax)
    
    def __call__(self, points):
        # calculate implicit values of points 
        ts = self.project_line(points)
        projs = self.interpolate(ts)

        dist = np.linalg.norm(points - projs, axis=1, ord=self.ord)
        return dist - self.r
