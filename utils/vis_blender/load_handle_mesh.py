import bpy
import os
import os.path as op

if 'Cube' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)

root_path = ''
idx = 0
shape_path = op.join(root_path , f'{idx}')
meshes_path = op.join(root_path, f'{idx}', 'handle', 'meshes')

col_default = bpy.data.collections['Collection']
if 'Handle' not in bpy.data.collections:
    col_handle = bpy.data.collections.new('Handle')
    bpy.context.scene.collection.children.link(col_handle)
else:
    col_handle = bpy.data.collections['Handle']


for file in os.listdir(meshes_path):
    name, ext = op.splitext(file)
    if not ext == '.ply':
        continue

    mesh_path = op.join(meshes_path, file)
    bpy.ops.import_mesh.ply(filepath=mesh_path)
    obj_mesh = bpy.data.objects[name]
    col_handle.objects.link(obj_mesh)
    col_default.objects.unlink(obj_mesh)