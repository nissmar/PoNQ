import os
import igl
import trimesh
import argparse
from tqdm import tqdm
import numpy as np
import joblib


def define_options_parser():
    parser = argparse.ArgumentParser(
        description='Test watertightness of the output shapes.')
    parser.add_argument('mesh_dir', type=str,
                        help='path to input meshes')
    return parser


def hash_max(a, b, N):
    return np.maximum(a+N*b, N*a+b)


def even_edge_index(v, f):
    edges = np.concatenate((hash_max(f[:, 0], f[:, 1], len(
        v)+1), hash_max(f[:, 0], f[:, 2], len(v)+1), hash_max(f[:, 1], f[:, 2], len(v)+1)))
    _, counts = np.unique(edges, return_counts=True)
    return ((counts % 2) == 0).all()


def check_single(src_dir, model_name):
    v, f = igl.read_triangle_mesh(src_dir + model_name)
    wrong_topology = 0
    wrong_geometry = 0
    empty = 0
    if len(v) > 0:
        wtop = not (even_edge_index(v, f))
        wgeo = int(os.popen(
            'src/cpp_utils/build/self_intersect {}{}'.format(src_dir, model_name)).read()[:-1])
        if wtop:
            wrong_topology = 1
        if wgeo > 0:
            wrong_geometry = 1
    else:
        empty = 1
    return (wrong_topology, wrong_geometry, empty, len(v), len(f))


def evaluate(src_dir):
    wrong_geometry = 0
    wrong_topology = 0
    empty = 0
    watertight = 0
    model_names = [model_name for model_name in os.listdir(src_dir) if ((".obj" in model_name or ".off" in model_name) and not (
        model_name[0] == '.' or '96481' in model_name or '58168' in model_name or 'poisson' in model_name))]
    # model_names = [model_name for model_name in os.listdir(src_dir) if ((".obj" in model_name or ".off" in model_name) and not (
    #     model_name[0] == '.' or '96481' in model_name or '58168' in model_name) and ("poisson" in model_name))]
    out = joblib.Parallel(n_jobs=-1)(joblib.delayed(check_single)
                                     (src_dir, name) for name in tqdm(model_names))
    out = np.array(out)
    wrong_topology = out[:, 0].sum()
    wrong_geometry = out[:, 1].sum()
    empty = out[:, 2].sum()
    watertight = np.logical_not(out[:, 0]+out[:, 1]).sum()
    mean_vertices = out[:, 3].mean()
    mean_faces = out[:, 4].mean()
    return (wrong_geometry, wrong_topology, empty, watertight / len(model_names), mean_vertices, mean_faces)


if __name__ == '__main__':
    parser = define_options_parser()
    args = parser.parse_args()
    scores = evaluate(args.mesh_dir+'/')
    print(args.mesh_dir)
    print('Percentage of watertight meshes: {}%'.format(scores[3]*100))
    print('Self-intersecting models: {}'.format(scores[0]))
    print('Non-closed models: {}'.format(scores[1]))
    print('Empty models: {}'.format(scores[2]))
    print('#V, #F (x10^3): {:.1f} & {:.1f}'.format(
        scores[4]/1000, scores[5]/1000))
