import bpy, bmesh
import os, glob, pickle
import os.path as op
from time import time

t0 = time()

if 'Cube' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)

def create_material(name, arg):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    # base color of P-BSDF
    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = arg['base_color']
    # metallic value
    mat.node_tree.nodes["Principled BSDF"].inputs[6].default_value = arg['metallic']
    # specular value
    mat.node_tree.nodes["Principled BSDF"].inputs[6].default_value = arg['specular']
    # roughness
    mat.node_tree.nodes["Principled BSDF"].inputs[9].default_value = arg['roughness']
    return mat

# Render setting
light = bpy.data.objects['Light']
light.location[0] = 1.28
light.location[1] = 1.1
light.location[2] = 6
light.rotation_euler[0] = 0.65
light.rotation_euler[1] = 0.
light.rotation_euler[0] = 1.92
cam = bpy.data.objects['Camera']
cam.location[0] = 2.3
cam.location[1] = 0.
cam.location[2] = 4.
cam.rotation_euler[0] = 0.
cam.rotation_euler[1] = 0.5236
cam.rotation_euler[2] = 0.
cam.data.lens = 80

bpy.context.scene.render.resolution_x = 1080
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.film_transparent = True
bpy.context.scene.view_settings.view_transform = 'Standard'

red = (1,0,0,1)
blue = (0,0,1,1)
cyan = (0., 0.8, 1., 1)
PBSDF_arg = {
    'base_color': cyan,
    'metallic': 0.1,
    'specular': 0.1,
    'roughness': 0.4,
}
mesh_mat = create_material('mesh', PBSDF_arg)


root_path = '.'
mesh_folder = op.join(root_path, '')
mesh_file = op.join(mesh_folder, '')
name, _ = op.splitext(op.basename(mesh_file))

bpy.ops.import_mesh.ply(filepath=mesh_file)
mesh = bpy.data.objects[name]
mesh.data.materials.append(mesh_mat)
mesh.active_material_index = 0
