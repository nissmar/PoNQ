import torch
from torch import nn
import numpy as np
import trimesh
import torch_scatter
from pytorch3d.ops import knn_points
from quadric_meshing import rvd_dual, SurfaceFromQuadrics


def vstars_from_quadrics(Q, P, eps=1e-2):
    """Q: Nx4x4 tensor of quadric matrices
       P: Nx3 tensor of positions
    returns vstars and eigen values"""
    A = Q[:, :3, :3]
    b = -Q[:, 3, :3]
    u, eigs, vh = torch.linalg.svd(A)

    eigs2 = torch.zeros_like(eigs)
    mask_s = eigs / eigs[:, 0, None] > eps
    eigs2[mask_s] = 1 / eigs[mask_s]

    base_pos = P
    vstars = base_pos + (
        vh.transpose(1, 2)
        @ torch.diag_embed(eigs2)
        @ u.transpose(1, 2)
        @ (b[..., None] - A @ base_pos[..., None])
    ).squeeze(-1)
    return vstars, vh, eigs


def cluster_samples_quadrics(predicted_points, samples, normals):
    """ Compute mean statistics in the voronoi region of predicted_points
    returns target_As, target_mean_normals, target_vstars, mask_exists (to apply to target_vstars)"""
    closest_idx = knn_points(
        samples, predicted_points).idx.squeeze(-1)  # k=1

    # Compute mean quadrics in the voronoi
    ds = -(normals * samples).sum(-1)
    ps = torch.cat((normals, ds[..., None]), -1)
    quadrics = (torch.matmul(ps[:, :, :, None], ps[:, :, None, :]))
    quadrics = quadrics.reshape(
        samples.shape[0], samples.shape[1], 16
    )
    mean_quadrics = torch_scatter.scatter_mean(
        quadrics,
        closest_idx.unsqueeze(2).repeat(1, 1, 16),
        1,
        torch.zeros(
            (predicted_points.shape[0], predicted_points.shape[1], 16), device=predicted_points.device),
    )
    mean_quadrics = mean_quadrics.reshape(
        predicted_points.shape[0], predicted_points.shape[1], 4, 4)

    # Compute mean normals
    mean_normals = torch_scatter.scatter_mean(
        normals,
        closest_idx.unsqueeze(2).repeat(1, 1, 3),
        1,
        torch.zeros(
            (predicted_points.shape[0], predicted_points.shape[1], 3), device=predicted_points.device),
    )
    # Compute count
    count = torch_scatter.scatter(
        torch.ones_like(closest_idx.unsqueeze(2)),
        closest_idx.unsqueeze(2),
        1,
        torch.zeros(
            (predicted_points.shape[0], predicted_points.shape[1], 1), device=predicted_points.device, dtype=torch.int64),
    )

    return mean_quadrics, mean_normals, count.squeeze(-1)


def closest_point(predicted_points, gt_points):
    pred_to_samples = knn_points(predicted_points, gt_points)
    return (pred_to_samples.dists).squeeze(-1)


class PoNQ(nn.Module):
    def __init__(self, points: np.ndarray, device):
        super().__init__()
        self.points = torch.tensor(
            points, dtype=torch.float32, requires_grad=True, device=device)
        self.mean_normals = torch.zeros(
            points.shape, dtype=torch.float32, device=device)
        self.quadrics = torch.zeros(
            (len(points), 10), dtype=torch.float32, device=device)
        self.non_void = torch.zeros(
            len(points), dtype=torch.bool, device=device)
        self.indices4x4 = torch.triu_indices(4, 4, device=device)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        with torch.no_grad():
            self.points = self.points.to(*args, **kwargs)
            self.mean_normals = self.mean_normals.to(*args, **kwargs)
            self.quadrics = self.quadrics.to(*args, **kwargs)
            self.non_void = self.non_void.to(*args, **kwargs)
            self.indices4x4 = self.indices4x4.to(*args, **kwargs)
        return self

    # Array manipulatoions
    def np_points(self, ret_q=False):
        if ret_q:
            return (
                self.points.cpu().detach().numpy(),
                self.quadrics.cpu().detach().numpy(),
            )
        return self.points.cpu().detach().numpy()

    def get_quadric_matrices(self):
        """get Nx4x4 tensor of quadric matrices"""
        Q = torch.zeros((len(self.points), 4, 4), device=self.points.device)
        Q[:, self.indices4x4[0], self.indices4x4[1]] = self.quadrics
        Q.transpose(1, 2)[:, self.indices4x4[0],
                          self.indices4x4[1]] = self.quadrics
        return Q

    def quadric_matrix_to_vector(self, bmat):
        return bmat[:, self.indices4x4[0], self.indices4x4[1]]

    # Losses
    def distance_to_centroids(self, points, centroids):
        return ((points[:, None, :] - centroids[None, ...]) ** 2).sum(-1)

    def cluster_samples_quadrics_normals(self, samples, normals, assign=True):
        closest_idx = knn_points(
            samples[None, :], self.points[None, :]).idx.squeeze()

        # sum of quadrics
        ds = -(normals * samples).sum(-1)
        ps = torch.cat((normals, ds[..., None]), -1)
        quadrics = (torch.matmul(ps[:, :, None], ps[:, None, :]))
        quadrics = quadrics.reshape(
            samples.shape[0], 16
        )
        target_quadrics = torch_scatter.scatter(
            quadrics,
            closest_idx,
            0,
            torch.zeros((len(self.points), 16), device=samples.device),
        )
        target_quadrics = target_quadrics.reshape(len(self.points), 4, 4)

        # mean normals
        mean_normals = torch_scatter.scatter_mean(
            normals,
            closest_idx,
            0,
            torch.zeros((len(self.points), 3), device=samples.device),
        )

        non_void = closest_idx.unique()

        if assign:
            with torch.no_grad():
                self.quadrics[:] = self.quadric_matrix_to_vector(
                    target_quadrics)
                self.mean_normals[:] = mean_normals
                self.non_void[:] = 0
                self.non_void[non_void] = 1
        else:
            return target_quadrics, mean_normals, non_void

    def get_vstars(self, eps=.05):
        """returns vstars, vectors and eigen values"""
        Q = self.get_quadric_matrices()[self.non_void]
        P = self.points[self.non_void]
        return vstars_from_quadrics(Q, P, eps)

    # mesh extraction
    def min_cut_surface(self, grid_scale, eps=.05, return_scores=False, correct_tet_color=True, open_treshold=None, return_indices=False):
        vstars, _, eigs = self.get_vstars(eps)
        SC = SurfaceFromQuadrics(vstars, eigs, self.get_quadric_matrices()[
            self.non_void], self.mean_normals[self.non_void],
            grid_scale=grid_scale,
            correct_tet_color=correct_tet_color,
            compute_mincut=True)
        return SC.get_surface(return_scores, open_treshold=open_treshold, return_indices=return_indices)

    # display

    def quadric_ellipse_mesh(self, size=0.02, subdivisions=1, lambd=1, color_func=None):
        vs = []
        fs = []
        cs = []
        vstars, vhs, eigs = self.get_vstars()
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
        mv = np.array(sphere.vertices)
        mf = np.array(sphere.faces)
        if not color_func is None:
            if len(color_func.shape) == 1:
                color_func = color_func[:, None]

        for i, (vstar, eig, vh) in enumerate(zip(
            vstars.cpu().detach().numpy(),
            eigs.cpu().detach().numpy(),
            vhs.cpu().detach().numpy(),
        )):
            if eig[0] > 0:
                eig /= np.sqrt((eig**2).sum(-1))
                trans = size * ((mv @ vh) * np.exp(-lambd * eig)) @ vh + vstar
                if len(trans) != len(mv):
                    print("err")
                fs.append(mf + len(mv) * len(vs))
                vs.append(trans)
                if color_func is None:
                    cs.append(np.ones(len(mf)) * (1-eig[1]/eig[0]))
                else:
                    cs.append(color_func[i][None, :].repeat(len(mf), 0))
        return np.concatenate(vs), np.concatenate(fs), np.concatenate(cs)

    def export_vstars(self, name, eps=1e-2):
        vstars, _, _ = self.get_vstars(eps)
        try:
            file = open(name, "x")
        except:
            file = open(name, "w")
        for e in vstars:
            file.write("{} {} {}\n".format(*e))
        file.write("\n")


# Neural network
class QuadricBaseNN(nn.Module):
    """to be used with neural networks"""

    def __init__(self):
        super().__init__()
        self.indices3x3 = torch.triu_indices(3, 3)  # 6 parameterss

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.indices3x3 = self.indices3x3.to(*args, **kwargs)
        return self

    def get_As(self, L_flat):
        """ L_flat: NxMx6
            returns: NxMx3x3 symetric matrix in cholesky decomposition LL^T"""
        L = torch.zeros(
            (*L_flat.shape[:-1], 3, 3), device=L_flat.device)
        L[..., self.indices3x3[0], self.indices3x3[1]] = L_flat
        return torch.matmul(L, L.transpose(-2, -1))

    def cluster_samples_quadrics(self, predicted_points, samples, normals):
        """ Compute mean statistics in the voronoi region of predicted_points
        returns target_As, target_mean_normals, target_vstars, mask_exists (to apply to target_vstars)"""
        closest_idx = knn_points(
            samples, predicted_points).idx.squeeze(-1)  # k=1

        # Compute mean quadrics in the voronoi
        ds = -(normals * samples).sum(-1)
        ps = torch.cat((normals, ds[..., None]), -1)
        quadrics = (torch.matmul(ps[:, :, :, None], ps[:, :, None, :]))
        quadrics = quadrics.reshape(
            samples.shape[0], samples.shape[1], 16
        )
        target_quadrics = torch_scatter.scatter_mean(
            quadrics,
            closest_idx.unsqueeze(2).repeat(1, 1, 16),
            1,
            torch.zeros(
                (predicted_points.shape[0], predicted_points.shape[1], 16), device=predicted_points.device),
        )
        target_quadrics = target_quadrics.reshape(
            predicted_points.shape[0], predicted_points.shape[1], 4, 4)

        # Compute mean normals
        mean_normals = torch_scatter.scatter_mean(
            normals,
            closest_idx.unsqueeze(2).repeat(1, 1, 3),
            1,
            torch.zeros(
                (predicted_points.shape[0], predicted_points.shape[1], 3), device=predicted_points.device),
        )

        non_void = (mean_normals**2).sum(-1) > 0

        target_As = target_quadrics[:, :, :3, :3]
        target_vstars, _, _ = vstars_from_quadrics(
            target_quadrics[non_void], predicted_points[non_void])
        return target_As, mean_normals, target_vstars, non_void

    def quadric_ellipse_mesh(self, predicted_vstars, predicted_As, size=0.02, subdivisions=1, lambd=1):
        # Draw mesh in quadric ellipse form
        vs = []
        fs = []
        cs = []

        _, eigs, vhs = torch.linalg.svd(predicted_As)
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
        mv = np.array(sphere.vertices)
        mf = np.array(sphere.faces)
        for vstar, eig, vh in zip(
            predicted_vstars.cpu().detach().numpy(),
            eigs.cpu().detach().numpy(),
            vhs.cpu().detach().numpy(),
        ):
            if eig[0] > 0:
                eig /= np.sqrt((eig**2).sum(-1))
                trans = size * ((mv @ vh) * np.exp(-lambd * eig)) @ vh + vstar
                if len(trans) != len(mv):
                    print("err")
                fs.append(mf + len(mv) * len(vs))
                vs.append(trans)
                cs.append(np.ones(len(mv)) * (eig[0]))
        return np.concatenate(vs), np.concatenate(fs), np.concatenate(cs)
