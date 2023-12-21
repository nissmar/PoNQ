import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import joblib
device = 'cuda'


def make_mask_close(sdf_input, grid_n):
    is_p = np.abs(sdf_input) < (2/grid_n*np.sqrt(3))
    sum_arrays = (is_p[:-1, :-1, :-1]*is_p[1:, :-1, :-1]*is_p[:-1, 1:, :-1]*is_p[:-1,
                  :-1, 1:]*is_p[1:, 1:, :-1]*is_p[1:, :-1, 1:]*is_p[:-1, 1:, 1:]*is_p[1:, 1:, 1:])
    return sum_arrays


# def make_gt_mask(samples: np.ndarray, grid_n: int):
#     """subselects voxels which collide with pointcloud"""
#     samples_low = np.floor((samples + 1) / 2 * grid_n).astype(np.int64)
#     mask = np.zeros((grid_n, grid_n, grid_n))
#     mask[samples_low[:, 0], samples_low[:, 1], samples_low[:, 2]] = 1
#     return mask.reshape((grid_n**3)) > 0


def get_item(src_dir, model_name, grid_n, compute_gt, indices):
    file = h5py.File(src_dir + model_name)
    # original SDF is in [-0.5, 0.5]^3
    sdf = 2 * file['{}_sdf'.format(grid_n-1)][:][None, :]

    mask = make_mask_close(
        2 * file['{}_sdf'.format(grid_n-1)][:], grid_n).reshape((grid_n-1)**3)

    if compute_gt:
        samples = 2 * file["pointcloud"][:][indices]
        samples_normals = file['normals'][:][indices]
        gt_mask = gt_mask = file['{}_mask'.format(
            grid_n-1)][:]
    else:
        samples = np.array([])
        samples_normals = np.array([])
        gt_mask = np.array([])
    return sdf, mask, samples, samples_normals, gt_mask, model_name


class ABCDataset_multiple(Dataset):
    def __init__(
        self,
        src_dir,
        names_set,
        grid_n=33,
        compute_gt=True,
        subsample=int(1e5),
        n_jobs=-1,
    ):
        self.compute_gt = compute_gt
        indices = np.random.choice(np.arange(int(1e6)), subsample)
        out = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(get_item)
                                             (src_dir, model_name, grid_n, compute_gt, indices) for model_name in tqdm(names_set))
        self.sdfs = [e[0] for e in out]
        self.masks = [e[1] for e in out]
        self.samples = [e[2] for e in out]
        self.samples_normals = [e[3] for e in out]
        self.gt_masks = [e[4] for e in out]
        self.names = [e[5] for e in out]

    def __getitem__(self, index):
        return (
            self.sdfs[index],
            self.masks[index],
            self.samples[index],
            self.samples_normals[index],
            self.gt_masks[index],
            self.names[index],
        )

    def __len__(self):
        return len(self.sdfs)
