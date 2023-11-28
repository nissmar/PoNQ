import numpy as np
import trimesh
from sklearn.neighbors import KDTree

# ef1_radius = 0.004
# ef1_dotproduct_threshold = 0.2
# ef1_threshold = 0.005
sample_num = 100000

f1_threshold = 0.003
# ef1_radius = 0.01
# ef1_dotproduct_threshold = 0.1
e_angle_treshold = 30
e_sampling_N = int(1e5)
ef1_threshold = 0.005
# ef1_threshold = 0.01


def uniform_edge_sampling(mesh, angle_treshold, N_sampling):
    sharp = mesh.face_adjacency_angles > np.radians(angle_treshold)
    sharp_edges = mesh.face_adjacency_edges[sharp]
    if len(sharp_edges) == 0:
        return np.array([])
    v = mesh.vertices

    edge_length = np.sqrt(
        ((v[sharp_edges[:, 1]]-v[sharp_edges[:, 0]])**2).sum(-1))
    selected_edges = np.random.choice(
        len(edge_length), N_sampling, p=edge_length/edge_length.sum())
    lambdas = np.random.rand(len(selected_edges))[:, None]
    sampled_points = v[sharp_edges[selected_edges][:, 1]] * \
        lambdas + v[sharp_edges[selected_edges][:, 0]]*(1-lambdas)
    return sampled_points


def get_cd_f1_nc(name, scale_gt, eval_normalization):
    idx = name[0]
    gt_obj_name = name[1]
    pred_obj_name = name[2]

    # load gt
    gt_mesh = trimesh.load(gt_obj_name)
    gt_mesh.vertices[:] /= scale_gt
    gt_points, gt_indexs = gt_mesh.sample(sample_num, return_index=True)
    gt_normals = gt_mesh.face_normals[gt_indexs]
    # load pred
    pred_mesh = trimesh.load(pred_obj_name, force='mesh')
    try:
        pred_mesh.vertices[:] = eval_normalization(
            pred_mesh.vertices)
        pred_points, pred_indexs = pred_mesh.sample(
            sample_num, return_index=True)
        # pred_points = eval_normalization(pred_points)
        pred_normals = pred_mesh.face_normals[pred_indexs]
    except:
        print('\n\n\n WARNING \n\n\n')
        pred_points = np.zeros((1, 3))
        pred_mesh = trimesh.Trimesh(pred_points, np.array([]))
        pred_normals = np.zeros((1, 3))

    # cd and nc and f1

    # from gt to pred
    pred_tree = KDTree(pred_points)
    dist, inds = pred_tree.query(gt_points, k=1)
    recall = np.sum(dist < f1_threshold) / float(len(dist))
    gt2pred_mean_cd1 = np.mean(dist)
    dist = np.square(dist)
    gt2pred_mean_cd2 = np.mean(dist)
    neighbor_normals = pred_normals[np.squeeze(inds, axis=1)]
    dotproduct = np.abs(np.sum(gt_normals*neighbor_normals, axis=1))
    gt2pred_nc = np.mean(dotproduct)

    # from pred to gt
    gt_tree = KDTree(gt_points)
    dist, inds = gt_tree.query(pred_points, k=1)
    precision = np.sum(dist < f1_threshold) / float(len(dist))
    pred2gt_mean_cd1 = np.mean(dist)
    dist = np.square(dist)
    pred2gt_mean_cd2 = np.mean(dist)
    neighbor_normals = gt_normals[np.squeeze(inds, axis=1)]
    dotproduct = np.abs(np.sum(pred_normals*neighbor_normals, axis=1))
    pred2gt_nc = np.mean(dotproduct)

    cd1 = gt2pred_mean_cd1+pred2gt_mean_cd1
    cd2 = gt2pred_mean_cd2+pred2gt_mean_cd2
    nc = (gt2pred_nc+pred2gt_nc)/2
    if recall+precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0

    if True:
        gt_edge_points = uniform_edge_sampling(
            gt_mesh, e_angle_treshold, e_sampling_N)

        pred_edge_points = uniform_edge_sampling(
            pred_mesh, e_angle_treshold, e_sampling_N)
        if len(pred_edge_points) == 0:
            pred_edge_points = np.zeros([486, 3], np.float32)
        if len(gt_edge_points) == 0:
            ecd1, ecd2 = 0, 0
            ef1 = 1
        else:
            # from gt to pred
            tree = KDTree(pred_edge_points)
            dist, inds = tree.query(gt_edge_points, k=1)
            erecall = np.sum(dist < ef1_threshold) / float(len(dist))
            gt2pred_mean_ecd1 = np.mean(dist)
            dist = np.square(dist)
            gt2pred_mean_ecd2 = np.mean(dist)

            # from pred to gt
            tree = KDTree(gt_edge_points)
            dist, inds = tree.query(pred_edge_points, k=1)
            eprecision = np.sum(dist < ef1_threshold) / float(len(dist))
            pred2gt_mean_ecd1 = np.mean(dist)
            dist = np.square(dist)
            pred2gt_mean_ecd2 = np.mean(dist)

            ecd1 = gt2pred_mean_ecd1+pred2gt_mean_ecd1
            ecd2 = gt2pred_mean_ecd2+pred2gt_mean_ecd2
            if erecall+eprecision > 0:
                ef1 = 2 * erecall * eprecision / (erecall + eprecision)
            else:
                ef1 = 0

        return idx, cd1, cd2, f1, nc, ecd2, ef1

    return idx, cd1, cd2, f1, nc, 0, 0
