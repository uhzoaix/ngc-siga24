import bpy
import os, pickle
import os.path as op
import numpy as np


def get_data(name):
    shape = bpy.data.objects[name]
    verts = np.asarray([[v.co.x, v.co.y, v.co.z] for v in shape.data.vertices])
    edges = np.asarray(shape.data.edge_keys)
    return { 
        'vertices': verts,
        'edges': edges,
    }

root_path = '.'
shape_name = ''
pose_name = ''

skeleton_path = op.join(root_path, shape_name, 'handle', 'std_skeleton.pkl')
with open(skeleton_path, 'rb') as f:
    sk = pickle.load(f)

res = {}
for name, val in sk.items():
    if name not in bpy.data.objects:
        continue
    
    new_sk = get_data(name)
    res[name] = new_sk

output_path = ''
os.makedirs(output_path, exist_ok=True)
output_file = op.join(output_path, f'{pose_name}.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(res, f)