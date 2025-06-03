import os, pickle
import numpy as np
import trimesh
import os.path as op
import pymeshlab as ml
from time import time
from tqdm.autonotebook import tqdm
from ngc.handle import Handle

def meshlab_shape_sampling(shape_path, num_samples, noise_scale):
    ms = ml.MeshSet()
    ms.load_new_mesh(shape_path)
    ms.generate_sampling_poisson_disk(samplenum=num_samples)
    mesh = ms.current_mesh()

    verts = mesh.vertex_matrix()
    vn = mesh.vertex_normal_matrix()
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)

    # add noise 
    noise = np.random.normal(0, noise_scale, size=verts.shape[0])
    verts_around = verts + noise[:, None]* vn
    
    verts = np.concatenate([verts, verts_around], axis=0)
    return verts

def meshlab_volumetric_sampling(shape_path, num_samples):
    ms = ml.MeshSet()
    ms.load_new_mesh(shape_path)
    ms.generate_sampling_volumetric(
        samplesurfradius = ml.Percentage(0.),
        samplevolnum = num_samples,
    )
    mesh = ms.mesh(1)
    verts = mesh.vertex_matrix()
    return verts

def bbox_volumetric_sampling(shape_path, num_samples):
    mesh = trimesh.load(shape_path, process=False)
    V = np.asarray(mesh.vertices)
    bmin, bmax = V.min(axis=0), V.max(axis=0)
    scales = bmax - bmin

    # [0,1]^3
    samples = np.random.rand(num_samples, 3)
    samples *= scales[None,:]
    samples += bmin
    return samples


def meshlab_SDF_eval(shape_path, samples):
    ms = ml.MeshSet()
    ms.load_new_mesh(shape_path)
    pc_mesh = ml.Mesh(samples)
    ms.add_mesh(pc_mesh, 'pc')

    ms.compute_scalar_by_distance_from_another_mesh_per_vertex()
    pc_mesh = ms.mesh(1)

    sdf_vals = pc_mesh.vertex_scalar_array()
    return sdf_vals

def export_handle_data(handle, graph_path, handle_path):
    handle.export_skeleton_mesh(handle_path)
    graph_data = handle.export_neural_graph()
    with open(op.join(graph_path, 'graph.pkl'), 'wb') as f:
        pickle.dump(graph_data, f)


def ngc_dataset(arg):
    root_path = arg['root_path']
    file_name = arg['file_name']
    n_surface_samples = arg['n_surface_samples']
    n_space_samples = arg['n_space_samples']

    # items = os.listdir(root_path)
    items = np.loadtxt(
        op.join(root_path, 'data.txt'), dtype=str).tolist()

    with tqdm(total=len(items)) as pbar:
        for name in items:
            item_path = op.join(root_path, f'{name}')
            shape_file = op.join(item_path, 'mesh.ply')
            handle_path = op.join(item_path, 'handle')
            handle_file = op.join(handle_path, 'std_handle.pkl')
            handle_mesh_file = op.join(handle_path, 'std_mesh.ply')
            output_path = op.join(item_path, 'train_data')
            os.makedirs(output_path, exist_ok=True)
            output_file = op.join(output_path, file_name)

            if op.exists(output_file):
                # print('Exists: ', item)
                pbar.update(1)
                continue

            handle = Handle()
            handle.load(handle_file)
            
            if not op.exists(handle_mesh_file):
                export_handle_data(handle, output_path, handle_path)

            surface_samples = meshlab_shape_sampling(
                shape_file, n_surface_samples, 0.01
            )
            space_samples = meshlab_volumetric_sampling(
                handle_mesh_file, n_space_samples
            )
            # space_samples = bbox_volumetric_sampling(
            #     handle_mesh_file, n_space_samples
            # )

            surface_data = handle.prepare_samples(surface_samples)
            surface_sdf = meshlab_SDF_eval(shape_file, surface_data['samples'])
            surface_data['sdf'] = surface_sdf

            space_data = handle.prepare_samples(space_samples)
            space_sdf = meshlab_SDF_eval(shape_file, space_data['samples'])
            space_data['sdf'] = space_sdf

            train_data = {
                'surface': surface_data,
                'space': space_data,
            }

            with open(output_file, 'wb') as f:
                pickle.dump(train_data, f)

            pbar.update(1)

    print('Done')

if __name__ == "__main__":
    # np.random.seed(2024)

    root_path = '/path/to/dataset'
    arg = {
        'root_path': root_path,
        'file_name': 'sdf_samples.pkl',
        'n_surface_samples' : 30000,
        'n_space_samples' : 50000,
    }
    ngc_dataset(arg)
