import numpy as np
import torch

class MixFunc():
    """docstring for MixFunc."""
    def __init__(self, arg):
        self.mode = arg['mode']
        if self.mode == 'TL_mix':
            # truncated linear mixing
            self.TL_seg = arg['seg']
            self.ts_reverse = arg['ts_reverse']
            self.weights_reverse = arg['weights_reverse']
            if 'offset' in arg:
                self.offset = arg['offset']
        elif self.mode == 'TL_sym':
            # truncated linear mixing with symmetric axis
            self.TL_seg = arg['seg']
            self.sym_axis = arg['sym_axis']
            self.ts_reverse = arg['ts_reverse']
            self.weights_reverse = arg['weights_reverse']
            self.clip_max = 1.
            if 'clip_max' in arg:
                self.clip_max = arg['clip_max']
        elif self.mode == 'part_stretch':
            self.t0 = arg['t0']
            self.t0_new = arg['t0_new']
        else:
            raise NotImplementedError

    def __call__(self, coords):
        if hasattr(self, self.mode):
            method = getattr(self, self.mode)
            return method(coords)
        else:
            raise NotImplementedError

    def TL_mix(self, coords):
        a, b = self.TL_seg
        if hasattr(self, 'offset'):
            weights = (coords + self.offset - a) / (b-a)
        else:
            weights = (coords - a) / (b - a)
            
        if isinstance(coords, np.ndarray):
            weights = np.clip(weights, a_min=0., a_max=1.)
        else:
            weights = torch.clip(weights, min=0., max=1.)

        if self.weights_reverse:
            weights = 1. - weights

        ts = coords
        if self.ts_reverse:
            ts = 1 - coords
        return ts, weights
    
    def TL_sym(self, coords):
        a, b = self.TL_seg
        sym = self.sym_axis
        if isinstance(coords, np.ndarray):
            vals = np.where(
                coords < sym,
                coords,
                2*sym - coords
            )
            weights = (vals - a) / (b - a)
            weights = np.clip(weights, a_min=0., a_max=self.clip_max)
        else:
            vals = torch.where(
                coords < sym,
                coords,
                2*sym - coords
            )
            weights = (vals - a) / (b - a)
            weights = torch.clip(weights, min=0., max=self.clip_max)

        if self.weights_reverse:
            weights = 1. - weights

        ts = coords
        if self.ts_reverse:
            ts = 1 - coords
        return ts, weights

    def part_stretch(self, coords):
        new_coords = np.zeros_like(coords)
        part1 = coords <= self.t0_new
        part2 = np.logical_not(part1)

        # [0,t0_new]->[0, t0]
        new_coords[part1] = coords[part1]* (self.t0 / self.t0_new)
        coef1 = (1 - self.t0) / (1 - self.t0_new)
        coef2 = (self.t0 - self.t0_new) / (1 - self.t0_new)
        # [t0_new, 1] -> [t0, 1]
        new_coords[part2] = coef1*coords[part2] + coef2
        return new_coords
    

def define_mix_func(config, weights_reverse):
    config['weights_reverse'] = weights_reverse
    return MixFunc(config)