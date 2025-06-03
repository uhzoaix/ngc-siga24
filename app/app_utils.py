import os, sys, pickle, yaml
import os.path as op
current_path = op.dirname(op.abspath(__file__))
sys.path.append(op.dirname(current_path))

import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import KDTree

import network
from ngc import Handle, CurveHandle, MCGrid
from blend_utils import *
from mix_utils import *

def load_model(device, log_path, checkpoint='final'):
    config_path = op.join(log_path, 'config.yaml')
    opt = yaml.safe_load(open(config_path))

    if checkpoint == 'final' or checkpoint == 'post':
        ckpt_name = f'model_{checkpoint}.pth'
    else:
        ckpt_name = 'model_epoch_%04d.pth' % int(checkpoint)

    model = network.define_model(opt['model'])
    checkpoint_path = op.join(log_path, f'checkpoints/{ckpt_name}')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, opt

def load_model_state_dict(log_path, checkpoint='final'):
    config_path = op.join(log_path, 'config.yaml')
    opt = yaml.safe_load(open(config_path))
    if checkpoint == 'final' or checkpoint == 'post':
        ckpt_name = f'model_{checkpoint}.pth'
    else:
        ckpt_name = 'model_epoch_%04d.pth' % int(checkpoint)

    checkpoint_path = op.join(log_path, f'checkpoints/{ckpt_name}')
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = network.define_model(opt['model'])
    return model, opt, state_dict

def load_handle(handle_path):
    handle = Handle()
    handle.load(handle_path)
    return handle

def load_yaml_file(yaml_file):
    # yaml file or the dict loaded
    if isinstance(yaml_file, str):
        return yaml.safe_load(open(yaml_file))
    else:
        return yaml_file

def create_grid_like(mc_grid):
    new_grid = MCGrid({
        'level': mc_grid.grid_config['level'],
        'reso': mc_grid.reso,
    })
    return new_grid

def create_grid(config):
    return MCGrid(config)


def copy_curve(handle, curve_name):
    curve = handle.curve_dict[curve_name]
    curve_arg = curve.export_data()
    curve_arg['idx'] = 0
    curve_copy = CurveHandle(curve_arg)
    return curve_copy

def curve_transform(arg):
    curve = arg['curve']
    anchor_idx = arg['anchor_idx']
    target_point = arg['origin']
    target_radius = arg['radius']

    anchor = curve.core.key_points[anchor_idx]
    radius = curve.core.key_radius[anchor_idx].max()
    curve.core.key_points += target_point - anchor
    curve.core.key_radius *= target_radius /radius
    curve.update()

def samples2mesh(samples, file_path):
    mesh = trimesh.Trimesh(samples, process=False)
    mesh.export(file_path)

def blend_global_cylinders(arg):
    handle = arg['handle']
    mc_grid = arg['mc_grid']
    sigma = arg['sigma']

    smooth = SmoothMaxMin(3, sigma)

    for curve in handle.curves:
        sdfs, kidx = curve.calc_cylinder_global_implicit(mc_grid, sigma)
        mc_grid.update_grid_func(sdfs, kidx, smooth.min)

    mesh = mc_grid.extract_mesh()
    return mesh


def chamfer_haus_dist(x, y):
    x_tree = KDTree(x)
    y_tree = KDTree(y)

    d_x, _ = y_tree.query(x)
    d_y, _ = x_tree.query(y)
    cd = np.mean(d_x**2) + np.mean(d_y**2)
    hd = (np.max(d_x) + np.max(d_y)) / 2.
    return cd,hd


def sdf2image(out_file, N, sdfs, mask, a_max=1.):
    vals = np.abs(sdfs)
    a_max = max(np.max(vals[mask]), a_max)
    vals /= a_max

    # vals = vals.reshape((N,N))
    # cmap = plt.cm.get_cmap('rainbow')
    cmap = mpl.colormaps['rainbow']
    img_inside = cmap(vals[mask])
    img = np.ones((sdfs.shape[0],4), dtype=img_inside.dtype)
    # transparent background
    img[:, 3] = 0.
    img[mask] = img_inside
    img = img.reshape((N,N,4))
    
    fig = plt.figure(figsize=(1,1), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, aspect='auto')
    fig.savefig(out_file, dpi=1000)


if __name__ == "__main__":
    sdfs = np.random.rand(256)
    N = 16
    mask = np.arange(256)
    img = sdf2image(N, sdfs, mask)
    test_file = 'test.png'
    fig = plt.figure(frameon=False)
    fig.set_size_inches(N,N)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect='auto')
    fig.savefig(test_file)
    
