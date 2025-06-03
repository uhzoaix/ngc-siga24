import bpy
import os,pickle
import os.path as op
import numpy as np


def create_curve(points, edges, name, collection):
    new_mesh = bpy.data.meshes.new(f'mesh_{name}')
    new_mesh.from_pydata(points, edges, [])
    new_mesh.update()

    new_object = bpy.data.objects.new(name, new_mesh)
    collection.objects.link(new_object)

if 'Cube' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)

root_path = '.'
name = ''
skeleton_path = op.join(root_path, name, 'handle', 'std_skeleton.pkl')
with open(skeleton_path, 'rb') as f:
    sk = pickle.load(f)

if 'Curves' not in bpy.data.collections:
    collection_curve = bpy.data.collections.new('Curves')
    bpy.context.scene.collection.children.link(collection_curve)
else:
    collection_curve = bpy.data.collections['Curves']

for name, val in sk.items():
    points = val['vertices']
    edges = val['edges']
    create_curve(points, edges, name, collection_curve)
