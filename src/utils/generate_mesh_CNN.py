import torch
import os
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from ABC_dataset import ABCDataset_multiple
import joblib
import mesh_tools as mt
from SDF_CNN import CNN_3d_multiple_split
from CNN_to_PoNQ_or_lite import CNN_to_PoNQ

def export_min_cut(name, grid_scale):
    V = torch.load(name, map_location='cpu')
    try:
        mt.export_obj(*V.min_cut_surface(grid_scale), name[:-3])
    except:
        mt.export_obj(np.array([]), np.array([]), name[:-3])


if __name__ == '__main__':
    with torch.no_grad():
        device = 'cuda'
        parser = argparse.ArgumentParser(
            description='evaluate CNN')
        parser.add_argument('config', type=str, help='Path to config file.')
        parser.add_argument('-dataset', type=str,
                            default="ABC", help='dataset')
        parser.add_argument('-grid_n', type=int, default=33, help='grid_n')
        parser.add_argument('-subd', type=int, default=0, help='grid_n')
        parser.add_argument('-n_jobs', type=int, default=-1,
                            help='number of jobs to run in parrallel')
        args = parser.parse_args()

        with open(args.config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        if args.dataset == "ABC":
            names = "src/eval/abc_ordered.txt"
            hdf5 = '/data/nmaruani/DATASETS/gt_Quadrics/'
            with open(names, "r") as f:
                val_names = [e[:-1] for e in f.readlines()]
                val_names = val_names[int(len(val_names)*0.8):]
        elif args.dataset == "Thingi":
            names = "src/eval/watertight_thingi32_obj_list.txt"
            hdf5 = "/data/nmaruani/DATASETS/gt_Thingi32_NDC_norm/"
            with open(names, "r") as f:
                val_names = [e[:-1] for e in f.readlines()]
                val_names = [e + ".hdf5" for e in val_names]
        else:
            raise ("Wrong dataset, must be ABC or Thingi")
        
        save_dir = '/data/nmaruani/RESULTS/Quadrics/{}_{}_{}/'.format(
            args.dataset, cfg["training"]["model_name"][5:-3], args.grid_n-1)
        if args.subd==1:
            save_dir = save_dir[:-1] + '_lite/'
        elif args.subd>1:
            save_dir = save_dir[:-1] + '_lite_{}/'.format(args.subd)

        print("Saving models in {}".format(save_dir))
        try:
            os.mkdir(save_dir)
        except:
            print('WARNING: overwriting files')

        M = CNN_3d_multiple_split(device=device)
        M.load_state_dict(torch.load(cfg["training"]["model_name"], map_location=device))
        M.to(device)
        M.eval()

        dataset = ABCDataset_multiple(
            hdf5, val_names, grid_n=args.grid_n, compute_gt=False)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False)
        for i, (sdfs, mask, _, _, _, name) in enumerate(tqdm(train_loader)):
            V = CNN_to_PoNQ(M, sdfs,  args.grid_n, mask, args.subd)
            if args.dataset == "ABC":
                torch.save(V, '{}/test_{}.pt'.format(save_dir, i))
            elif args.dataset == "Thingi":
                torch.save(
                    V, '{}/{}.pt'.format(save_dir, name[0][:-5]))
        names = [save_dir+e for e in os.listdir(save_dir) if '.pt' in e]
        out = joblib.Parallel(
            n_jobs=args.n_jobs)(joblib.delayed(export_min_cut)(name, (args.grid_n-1)/2**args.subd) for name in tqdm(names))
