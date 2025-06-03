import bpy
import os,pickle, json
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

root_path = ''
name = ''
item_path = op.join(root_path, name)
mesh_file = op.join(item_path, 'mesh.ply')
handle_mesh_file = op.join(item_path, 'handle', 'std_mesh.ply')
bpy.ops.import_mesh.ply(filepath=mesh_file)
bpy.ops.import_mesh.ply(filepath=handle_mesh_file)

skeleton_path = op.join(item_path, 'handle', 'handle_setting.pkl')
with open(skeleton_path, 'rb') as f:
    sk = pickle.load(f)

if 'Curves' not in bpy.data.collections:
    collection_curve = bpy.data.collections.new('Curves')
    bpy.context.scene.collection.children.link(collection_curve)
else:
    collection_curve = bpy.data.collections['Curves']

for name, val in sk.items():
    points = val['points']
    num = points.shape[0]
    pid = np.arange(num)
    edges = np.asarray([pid[:-1], pid[1:]]).T
    create_curve(points, edges, name, collection_curve)
