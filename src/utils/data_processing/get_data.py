import numpy as np
import os
import h5py
from multiprocessing import Process, Queue
import time
import trimesh
import joblib
from tqdm import tqdm
import utils

def get_gt_from_intersectionpn(name_list):
    cell_voxel_size = 8
    num_of_int_params = 3
    num_of_float_params = 3

    point_sample_num = int(1e6)

    grid_size_list = [32, 64, 128]
    LOD_gt_int = {}
    LOD_gt_float = {}
    LOD_input_sdf = {}
    LOD_input_voxel = {}
    for grid_size in grid_size_list:
        grid_size_1 = grid_size+1
        LOD_gt_int[grid_size] = np.zeros(
            [grid_size_1, grid_size_1, grid_size_1, num_of_int_params], np.uint8)
        LOD_gt_float[grid_size] = np.zeros(
            [grid_size_1, grid_size_1, grid_size_1, num_of_float_params], np.float32)
        LOD_input_sdf[grid_size] = np.ones(
            [grid_size_1, grid_size_1, grid_size_1], np.float32)
        LOD_input_voxel[grid_size] = np.zeros(
            [grid_size_1, grid_size_1, grid_size_1], np.uint8)

    in_name = name_list[2]
    out_name = name_list[3]

    in_obj_name = in_name + ".obj"
    in_sdf_name = in_name + ".sdf"
    out_hdf5_name = out_name + ".hdf5"
    command = "./SDFGen "+in_obj_name+" 128 0"
    os.system(command)

    # read
    gt_mesh = trimesh.load(in_obj_name)
    gt_points, face_idx = trimesh.sample.sample_surface(
        gt_mesh, point_sample_num)
    gt_normals = np.array(gt_mesh.face_normals[face_idx])

    sdf_129 = utils.read_sdf_file_as_3d_array(in_sdf_name)  # 128

    # compute gt
    for grid_size in grid_size_list:
        grid_size_1 = grid_size+1
        voxel_size = grid_size*cell_voxel_size
        downscale = 1024//voxel_size
        # prepare downsampled voxels and intersections
        tmp_sdf = sdf_129[0::downscale, 0::downscale, 0::downscale]
        LOD_input_sdf[grid_size][:] = tmp_sdf

    # record data
    hdf5_file = h5py.File(out_hdf5_name, 'w')
    hdf5_file.create_dataset(
        "pointcloud", [point_sample_num, 3], np.float32, compression=9)
    hdf5_file["pointcloud"][:] = gt_points

    hdf5_file.create_dataset(
        "normals", [point_sample_num, 3], np.float32, compression=9)
    hdf5_file["normals"][:] = gt_normals

    for grid_size in grid_size_list:
        grid_size_1 = grid_size+1
        hdf5_file.create_dataset(str(
            grid_size)+"_sdf", [grid_size_1, grid_size_1, grid_size_1], np.float32, compression=9)
        hdf5_file[str(grid_size)+"_sdf"][:] = LOD_input_sdf[grid_size]
    hdf5_file.close()
    os.remove(in_sdf_name)


if __name__ == '__main__':
    target_dir = "/data/nmaruani/DATASETS/ABC/"
    if not os.path.exists(target_dir):
        print("ERROR: this dir does not exist: "+target_dir)
        exit()

    write_dir = "/data/nmaruani/DATASETS/gt_Quadrics/"
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
        
    obj_names = os.listdir(target_dir)
    obj_names = sorted(obj_names)

    obj_names_len = len(obj_names)

    list_of_names = []
    for idx in range(obj_names_len):
        in_name = target_dir + obj_names[idx] + "/model"
        out_name = write_dir + obj_names[idx]
        list_of_names.append(
            [0, idx, in_name, out_name])
    joblib.Parallel(n_jobs=-1)(joblib.delayed(get_gt_from_intersectionpn)
                               (name) for name in (tqdm(list_of_names)))
   