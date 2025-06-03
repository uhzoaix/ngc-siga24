import os, sys, pickle
import torch
import numpy as np
import trimesh

import os.path as op
from torch.utils.data import Dataset
from ngc import Handle

class NGCDataset(Dataset):
    """
    docstring for NGCDataset.

    """
    def __init__(self, arg):
        super(NGCDataset, self).__init__()
        self.root_path = arg['root']

        self.mode = arg['mode']
        if self.mode == 'train':
            self.n_sample = arg['n_sample']
            self.data_names = np.loadtxt(
                op.join(self.root_path, 'data.txt'), dtype=str).tolist()
            
            if 'shape_name' in arg:
                self.data_names = [str(arg['shape_name'])]

            self.file_name = 'sdf_samples.pkl'

            self.handles = self.load_handles()
            self.inputs = self.load_inputs()
    
    def load_handles(self):
        handles = []
        for name in self.data_names:
            item_path = op.join(self.root_path, name)
            handle_path = op.join(item_path, 'handle', 'std_handle.pkl')
            handle = Handle()
            handle.load(handle_path)
            handles.append(handle)

        return handles
    
    def load_inputs(self):
        inputs = []
        curve_nums = [handle.num_curve for handle in self.handles]
        print('Total num of curves:{}, {} shapes'.format(
            sum(curve_nums), len(curve_nums)
        ))
        curve_nums.insert(0, 0)
        idx_range = np.cumsum(curve_nums)
        hid = 0

        for name in self.data_names:
            start, end = idx_range[hid], idx_range[hid+1]
            inputs.append(np.arange(start, end))
            hid += 1
            
        return inputs


    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx):
        name = self.data_names[idx]
        item_path = op.join(self.root_path, name)
        data_path = op.join(item_path, 'train_data', self.file_name)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        input_curve_idx = self.inputs[idx]
        input1, gt1 = self.get_curve_data(data['surface'], input_curve_idx)
        input2, gt2 = self.get_curve_data(data['space'], input_curve_idx)
        data_input = {
            'samples': torch.cat([input1['samples'], input2['samples']], dim=0),
            'coords': torch.cat([input1['coords'], input2['coords']]),
            'curve_idx': torch.cat([input1['curve_idx'], input2['curve_idx']]),
        }
        data_gt = {
            'sdf': torch.cat([gt1['sdf'], gt2['sdf']]),
        }
        info = {}

        return data_input, data_gt, info

    def get_curve_data(self, curve_data, input_curve_idx):
        samples_local = curve_data['samples_local']
        samples_coords = curve_data['coords']
        samples_sdf = curve_data['sdf']

        cids = curve_data['curve_idx'].astype(np.int32)

        num_samples = samples_local.shape[0]
        if self.n_sample <= 0:
            sidx = np.arange(num_samples)
        else:
            n_s = self.n_sample
            if n_s <= num_samples:
                sidx = np.random.choice(num_samples, size=n_s, replace=False)
            else:
                raise ValueError(f'num of samples{num_samples} smaller than the threshold {n_s}')
        
        gt_sdf = samples_sdf[sidx]

        samples_local = samples_local[sidx]
        samples_coords = samples_coords[sidx]
        cids = cids[sidx]
        curve_idx = input_curve_idx[cids]
        model_input = {
            'samples': torch.from_numpy(samples_local).float(),
            'coords': torch.from_numpy(samples_coords).float(),
            'curve_idx': torch.from_numpy(curve_idx).long(),
        }
        gt = {
            'sdf': torch.from_numpy(gt_sdf).float(),
        }
        return model_input, gt
    

class DeepSDFDataset(Dataset):
    """
    docstring for DeepSDFDataset.

    """
    def __init__(self, arg):
        super(DeepSDFDataset, self).__init__()
        self.root_path = arg['root']

        self.mode = arg['mode']
        if self.mode == 'train':
            self.n_sample = arg['n_sample']
            self.data_names = np.loadtxt(
                op.join(self.root_path, 'data.txt'), dtype=str).tolist()

            self.file_name = 'sdf_samples.pkl'

    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx):
        name = self.data_names[idx]
        item_path = op.join(self.root_path, name)
        data_path = op.join(item_path, 'train_data', self.file_name)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        input1, gt1 = self.get_data(data['surface'])
        input2, gt2 = self.get_data(data['space'])
        data_input = {
            'samples': torch.cat([input1['samples'], input2['samples']], dim=0),
            'idx': torch.LongTensor([idx]),
        }
        data_gt = {
            'sdf': torch.cat([gt1['sdf'], gt2['sdf']]),
        }
        info = {}

        return data_input, data_gt, info

    def get_data(self, data):
        samples = data['samples']
        samples_sdf = data['sdf']

        num_samples = samples.shape[0]
        if self.n_sample <= 0:
            sidx = np.arange(num_samples)
        else:
            n_s = self.n_sample
            if n_s <= num_samples:
                sidx = np.random.choice(num_samples, size=n_s, replace=False)
            else:
                raise ValueError(f'num of samples{num_samples} smaller than the threshold {n_s}')
        
        samples = samples[sidx]
        gt_sdf = samples_sdf[sidx]
        model_input = {
            'samples': torch.from_numpy(samples).float(),
        }
        gt = {
            'sdf': torch.from_numpy(gt_sdf).float(),
        }
        return model_input, gt