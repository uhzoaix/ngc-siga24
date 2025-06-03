import os, pickle, yaml
import os.path as op
import torch
import numpy as np
from time import time
from utils import MCGrid
from app import Agent

t0 = time()
# seed = 2025
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')

agent = Agent()

root_path = op.dirname(op.abspath(__file__))
exp_name = 'demo'
log_path = op.join(root_path, 'results', exp_name)
data_root = op.join(log_path, 'dataset')

output_path = op.join(root_path, 'inference', exp_name)
os.makedirs(output_path, exist_ok=True)

agent.load_model(device, log_path, checkpoint='final')
agent.load_data(data_root)
print('Model and handle data loaded, time cost: ', time()-t0)

# Marching Cubes config
config_path = './exp/demo/manipulation'
grid_config = {
    'reso': 256,
    'level': 0.
}

t0 = time()
mc_grid = MCGrid(grid_config)
shape_name = '222'
arg = {
    'exp_name': 'mix',
    'data_root': data_root, 
    'mc_grid': mc_grid,
    'output_folder': op.join(output_path, f'{shape_name}'),
    'shape': shape_name,
    'mixing_file': op.join(config_path, f'mix_{shape_name}.yaml'),
}
agent('part_mixing', arg)
print('time cost: ', time()-t0)

t0 = time()
mc_grid = MCGrid(grid_config)
shape_name = '388'
arg = {
    'exp_name': 'transform',
    'data_root': data_root, 
    'mc_grid': mc_grid,
    'output_folder': op.join(output_path, f'{shape_name}'),
    'shape': shape_name,
    'transform_file': op.join(config_path, f'transform_{shape_name}.yaml'),
}
agent('shape_transform', arg)
print('time cost: ', time()-t0)



# generate interpolated meshes for videos

#-------------------------
# # for part mixing
#-------------------------

# num_frame = 36
# mix_file = arg['mixing_file']
# video_name = arg['exp_name']
# min_offset, max_offset = 0., 1.
# diff = max_offset - min_offset
# for i in range(num_frame):
#     grid_config = {
#         'reso': 256,
#         'level': 0.
#     }
#     mc_grid = MCGrid(grid_config)

#     val = i / (num_frame - 1)
#     config = yaml.safe_load(open(mix_file))
#     offset_frame = min_offset + val*diff
#     # config['248|L_wing']['offset'] = offset_frame
#     # config['248|R_wing']['offset'] = offset_frame
#     config['002946|body']['clip_max'] = offset_frame
#     print(f'Frame{i}: {offset_frame}')

#     arg['mixing_file'] = config
#     arg['exp_name'] = f'{video_name}_{i:03}'
#     arg['mc_grid'] = mc_grid
#     agent('part_mixing', arg)


#-------------------------
# # for scaling
#-------------------------

# num_frame = 36
# transform_file = arg['transform_file']
# video_name = arg['exp_name']
# min_scale, max_scale = 1, 3
# diff = max_scale - min_scale
# for i in range(num_frame):
#     grid_config = {
#         'reso': 256,
#         'level': 0.
#     }
#     mc_grid = MCGrid(grid_config)

#     val = i / (num_frame - 1)
#     config = yaml.safe_load(open(transform_file))
#     scale_frame = min_scale + val*diff
#     config['scaling']['body']['scales'][1] = scale_frame
#     print(f'Frame{i}: {scale_frame}')

#     arg['transform_file'] = config
#     arg['exp_name'] = f'{video_name}_{i:03}'
#     arg['mc_grid'] = mc_grid
#     agent('shape_transform', arg)

#-------------------------
# # for twisting
#-------------------------
# num_frame = 36
# transform_file = arg['transform_file']
# video_name = arg['exp_name']
# min_angle, max_angle = 0, 180
# diff = max_angle - min_angle
# for i in range(num_frame):
#     grid_config = {
#         'reso': 256,
#         'level': 0.
#     }
#     mc_grid = MCGrid(grid_config)

#     val = i / (num_frame - 1)
#     config = yaml.safe_load(open(transform_file))
#     angle_frame = min_angle + val*diff
#     config['tilt']['shape']['angles'][1] = angle_frame
#     print(f'Frame{i}: {angle_frame}')

#     arg['transform_file'] = config
#     arg['exp_name'] = f'{video_name}_{i:03}'
#     arg['mc_grid'] = mc_grid
#     agent('shape_transform', arg)


#-------------------------
# for curve manipulation
#-------------------------
# num_frame = 36
# video_name = arg['exp_name']
# pi2 = 2*np.pi
# num_curvepts = 36
# val_scale = 0.25
# t_val = 0.5
# ts = np.linspace(-t_val, t_val, num=num_curvepts, endpoint=True)

# arg['shape'] = '222'
# arg['output_folder'] = op.join(output_path, '222')

# for i in range(num_frame):
#     grid_config = {
#         'reso': 128,
#         'level': 0.
#     }
#     mc_grid = MCGrid(grid_config)

#     val = i / (num_frame - 1)
#     offset = pi2 *val
#     sin_ts = pi2*(ts + t_val)/(2*t_val) + offset
#     y_val = val_scale*np.sin(sin_ts)
#     z_val = ts
#     points = np.zeros((num_curvepts, 3))
#     points[:,1] = y_val
#     points[:,2] = z_val
#     z_axis = np.asarray([1,0,0])

#     config = {
#         'body': {
#             'vertices': points
#         }
#     }
#     print(f'Frame{i}: {offset}')

#     arg['transform_file'] = {
#         'pose': {
#             'pose_file': config,
#             'z_axis': z_axis, 
#         },
#         'scaling' : {
#             'body': {
#                 'scales': [0.5, 0.5],
#                 'coords': [0, 1]
#             }
#         }
#     }
#     arg['exp_name'] = f'{video_name}_{i:03}'
#     arg['mc_grid'] = mc_grid
#     agent('shape_transform', arg)
    
