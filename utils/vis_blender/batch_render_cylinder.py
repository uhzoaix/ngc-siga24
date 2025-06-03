import bpy, bmesh
import os, glob, pickle
import os.path as op
import numpy as np
from time import time

t0 = time()

if 'Cube' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)

def create_cylinder_material(name, arg):
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
    # Alpha 
    mat.node_tree.nodes["Principled BSDF"].inputs[21].default_value = 0.4
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'NONE'
    return mat

def create_curve_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    shader = nodes.new('ShaderNodeEmission')
    nodes["Emission"].inputs[0].default_value = color
    links.new(shader.outputs[0], output.inputs[0])
    return mat

def load_curve(curve_path, _mat):
    name, ext = op.splitext(os.path.basename(curve_path))

    if not os.path.exists(curve_path):
        print('Not exists: ', curve_path)
        return

    with open(curve_path, 'rb') as f:
        curve_data = pickle.load(f)
        data = curve_data['body']

    pts = data['vertices']
    edges = data['edges']

    new_mesh = bpy.data.meshes.new(f'mesh_{name}')
    new_mesh.from_pydata(pts, edges, [])
    new_mesh.update()

    new_object = bpy.data.objects.new(f'object_{name}', new_mesh)
    scene = bpy.context.scene
    scene.collection.objects.link(new_object)
    bpy.context.view_layer.objects.active = new_object

    new_object.select_set(True)
    bpy.ops.object.convert(target='CURVE')

    new_object.data.bevel_depth = 0.008

    new_object.data.materials.append(_mat)
    new_object.active_material_index = 0

    return new_object


def load_curves(curve_path, mat):
    name, ext = op.splitext(os.path.basename(curve_path))
    with open(curve_path, 'rb') as f:
        curve_data = pickle.load(f)

    objs = []
    for part, data in curve_data.items():
        part_name = f'{name}_{part}'
        pts = data['vertices']
        edges = data['edges']

        new_mesh = bpy.data.meshes.new(f'mesh_{part_name}')
        new_mesh.from_pydata(pts, edges, [])
        new_mesh.update()

        new_object = bpy.data.objects.new(f'object_{part_name}', new_mesh)
        scene = bpy.context.scene
        scene.collection.objects.link(new_object)
        bpy.context.view_layer.objects.active = new_object

        new_object.select_set(True)
        bpy.ops.object.convert(target='CURVE')

        new_object.data.bevel_depth = 0.008

        new_object.data.materials.append(mat)
        new_object.active_material_index = 0

        new_object.rotation_euler[0] = -np.pi/2
        objs.append(new_object)
    
    return objs

def load_mesh(mesh_file, mat):
    name, _ = op.splitext(op.basename(mesh_file))
    bpy.ops.import_mesh.ply(filepath=mesh_file)
    mesh = bpy.data.objects[name]
    mesh.data.materials.append(mat)
    mesh.active_material_index = 0

    mesh.rotation_euler[0] = -np.pi/2
    return mesh

def get_out_name(mesh_file):
    name, _ = op.splitext(op.basename(mesh_file))
    splits = name.split('_')
    idx = int(splits[1])
    out_name = '{}_{:03}'.format(splits[0], idx)
    return out_name

# mesh.rotation_euler[0] = -1.5707
# mesh.rotation_euler[1] = -1.5707
    
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
cyan = (0., 1, 1., 1)
PBSDF_arg = {
    'base_color': cyan,
    'metallic': 0.,
    'specular': 0.,
    'roughness': 0.4,
}
mesh_mat = create_cylinder_material('mesh', PBSDF_arg)
curve_mat = create_curve_material('curve', red)

root_path = ''
output_path = ''


exp_name = 'cylinders'
img_folder = op.join(output_path, exp_name)
os.makedirs(img_folder, exist_ok=True)

names = ['']

# for mesh_file in glob.glob(op.join(mesh_folder, '*.ply')):
for name in names:
    # curve_file = mesh_file.replace('mesh.ply', 'skeleton.pkl')
    cyl_file = op.join(root_path, name, 'handle/std_mesh.ply')
    curve_file = op.join(root_path, name, 'handle/std_skeleton.pkl')
    mesh_obj = load_mesh(cyl_file, mesh_mat)
    curves = load_curves(curve_file, curve_mat)

    output_path = os.path.join(img_folder, f'{name}.png')
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(mesh_obj, do_unlink=True)
    # bpy.data.objects.remove(curve_obj, do_unlink=True)
    for cobj in curves:
        bpy.data.objects.remove(cobj, do_unlink=True)
