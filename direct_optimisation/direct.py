import open3d as o3d
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
import torch
import numpy as np
import os
import argparse
import yaml
from pytorch3d.ops import knn_points
from time import time

if True:
    import sys
    sys.path.insert(1, '../utils/')
    sys.path.insert(1, 'utils/')
    import mesh_tools as mt
    import neural_quadrics as nq


def initialize_model(n_points, device, input_pc=None, grid_n=0):
    '''uniform in [-.5, .5]^3 or farthest point sampling'''
    if grid_n != 0:
        points = mt.mesh_grid(grid_n, True)[
            mt.mask_relevant_voxels(grid_n, input_pc)]
    elif not (input_pc is None):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_pc))
        points = np.asarray(pcd.farthest_point_down_sample(n_points).points)
    else:
        points = 2*np.random.rand(n_points, 3)-1.

    return nq.MovingQuadrics(points, device)


def train_simple(V, optimizer, tensor_surface, repulsion_fac=0, sample_fac=1):
    optimizer.zero_grad()
    masks = torch.rand_like(tensor_surface[:, 0]) < sample_fac
    loss = chamfer_distance(
        tensor_surface[masks][None, :], V.points[None, :])[0].mean()
    if repulsion_fac > 0:
        loss += -repulsion_fac * \
            (knn_points(V.points[None, :],
             V.points[None, :], K=2).dists[0, :, 1].mean())
    x = loss.item()
    loss.backward()
    optimizer.step()
    return x


def save_output(V, out_dir, name, out_pt=False, out_rvd=False, out_poisson=False, out_min_cut_grid_size=False):
    if out_pt:
        torch.save(V, '{}/{}.pt'.format(out_dir, name))
    if out_rvd:
        mt.export_obj(
            *V.rvd_dual(), '{}/{}_rvd_dual'.format(out_dir, name))
    if out_poisson:
        ppoints = V.points[V.non_void].cpu().detach().numpy()
        pnormals = V.mean_normals[V.non_void].cpu().detach().numpy()

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ppoints))
        pcd.normals = o3d.utility.Vector3dVector(pnormals)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd)
        psv = np.asarray(mesh.vertices)
        psf = np.asarray(mesh.triangles)

        mt.export_obj(psv, psf, '{}/{}_poisson'.format(out_dir, name))
    if out_min_cut_grid_size:
        t0 = time()
        mcv, mcf = V.min_cut_surface(out_min_cut_grid_size)
        final_t = time() - t0
        mt.export_obj(
            mcv, mcf, '{}/{}'.format(out_dir, name))
        return final_t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Direct optimization experiment')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('-grid_n', type=int, default=None, help='grid size')

    args = parser.parse_args()
    TOTAL_OPTIM_TIME = 0
    TOTAL_MESH_TIME = 0
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    if not args.grid_n is None:
        cfg['io']['grid_n'] = args.grid_n
        cfg['path']['out_dir'] = cfg['path']['out_dir'].format(
            cfg['io']['grid_n'])
        cfg['io']['sample_n'] = int(
            cfg['io']['sample_n_base'] * (cfg['io']['grid_n']/32)**2)
        cfg['optim']['lr'] = cfg['optim']['lr_base']/(cfg['io']['grid_n']/32)
    try:
        os.mkdir(cfg['path']['out_dir'])
    except:
        print('WARNING: overwriting current files')

    for name in tqdm(os.listdir(cfg['path']['src_dir'])):
        model_path = cfg['path']['src_dir'] + name

        input_points, input_normals = mt.load_shape(
            model_path, cfg['io']['in_pointcloud'], cfg['io']['normalize'], cfg['io']['sample_n'])

        if cfg['model']['init'] == 'farthest' or not args.grid_n is None:
            input_pc = input_points
        else:
            input_pc = None
        t0 = time()
        V = initialize_model(cfg['model']['n_points'],
                             cfg['optim']['device'], input_pc, cfg['io']['grid_n'])
        optimizer = torch.optim.Adam([V.points], cfg['optim']['lr'])
        tensor_surface = torch.tensor(
            input_points, dtype=torch.float32).to(cfg['optim']['device'])
        tensor_normals = torch.tensor(
            input_normals, dtype=torch.float32).to(cfg['optim']['device'])
        for _ in range(cfg['optim']['epochs']):
            train_simple(
                V, optimizer, tensor_surface, repulsion_fac=cfg['optim']['repulsion_fac'], sample_fac=cfg['optim']['sample_fac'])
        # RVD dual Mesh
        V.cluster_samples_quadrics_normals(tensor_surface, tensor_normals)
        TOTAL_OPTIM_TIME += time()-t0
        # Save output
        out_min_cut_grid_size = cfg['io']['grid_n'] if not args.grid_n is None else False
        TOTAL_MESH_TIME += save_output(V, cfg['path']['out_dir'], name[:-4], cfg['io']
                                       ['out_pt'], cfg['io']['out_rvd'], cfg['io']['out_poisson'], out_min_cut_grid_size)
    print('TOTAL OPTIM TIME: ', TOTAL_OPTIM_TIME /
          len(os.listdir(cfg['path']['src_dir'])))
    print('TOTAL MESH TIME: ', TOTAL_MESH_TIME /
          len(os.listdir(cfg['path']['src_dir'])))
