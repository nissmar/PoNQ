import torch
import argparse
import yaml
from dataset.ABC_multiple import ABCDataset_multiple
from networks.SDF_CNN import CNN_3d_multiple_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pytorch3d.ops import knn_points, knn_gather
if True:
    import sys
    sys.path.insert(1, '../utils/')
    import neural_quadrics as nq
    from quadric_meshing import quadrics_score

BCE = torch.nn.BCELoss()


def chamfer_quadric_normal_loss(predicted_points, predicted_vstars, predicted_normals, predicted_quadrics, gt_points, gt_normals):
    pred_to_samples = knn_points(predicted_points, gt_points)
    samples_to_pred = knn_points(gt_points, predicted_points)
    # Compute mean quadrics in the voronoi
    gt_quadrics = (torch.matmul(
        gt_normals[:, :, :, None], gt_normals[:, :, None, :]))
    gt_quadrics = gt_quadrics.view(
        gt_quadrics.shape[0], gt_quadrics.shape[1], 9
    )
    predicted_quadrics = predicted_quadrics.view(
        predicted_quadrics.shape[0], predicted_quadrics.shape[1], 9)
    closest_quadric = knn_gather(
        predicted_quadrics, samples_to_pred.idx).squeeze(-2)
    closest_normal = knn_gather(
        predicted_normals, samples_to_pred.idx).squeeze(-2)
    closest_to_sample = knn_gather(
        predicted_vstars, samples_to_pred.idx).squeeze(-2)

    d1 = (pred_to_samples.dists).squeeze(-1).mean(1)
    d2 = (samples_to_pred.dists).squeeze(-1).mean(1)

    loss_chamfer = (d1 + d2).mean()
    loss_quadric = ((closest_quadric-gt_quadrics)**2).mean()
    loss_normals = ((closest_normal-gt_normals)**2).mean()

    loss_vstars = (((closest_to_sample-gt_points)*gt_normals).sum(-1)
                   ** 2).mean()
    loss_reg = ((predicted_points.detach()-predicted_vstars)**2).mean()
    return loss_chamfer, loss_vstars, loss_normals, loss_quadric, loss_reg


def train_single(p_points, p_vstars, p_normals, p_As, p_bool, gt_points, gt_normals, gt_bool):
    loss_chamfer, loss_vstars, loss_normals, loss_quadric, loss_reg = chamfer_quadric_normal_loss(
        p_points[None, :], p_vstars[None, :], p_normals[None, :], p_As[None, :], gt_points[None, :], gt_normals[None, :])
    loss_bools = (torch.relu((p_bool-1.0*gt_bool)**2-1e-2)).mean()
    return torch.stack((loss_chamfer, loss_vstars, loss_normals, loss_quadric, loss_bools, loss_reg))


def train(M, optimizer, train_loader, losses_weights, device='cuda'):
    loss_items = []
    for (sdfs, mask, samples, samples_n, gt_mask, names) in train_loader:
        optimizer.zero_grad()
        loss = 0
        loss_t = 0
        sdfs, samples, samples_n, mask, gt_mask = sdfs.to(device), samples.to(
            device), samples_n.to(device), mask.to(device), gt_mask.to(device)
        predicted_points, predicted_vstars, predicted_mean_normals, predicted_As, predicted_bool = M(
            sdfs)
        for i in range(sdfs.shape[0]):
            losses = train_single(predicted_points[i, gt_mask[i]].view(-1, 3),
                                  predicted_vstars[i, gt_mask[i]].view(-1, 3),
                                  predicted_mean_normals[i,
                                                         gt_mask[i]].view(-1, 3),
                                  predicted_As[i, gt_mask[i]].view(-1, 3, 3),
                                  predicted_bool[i, mask[i]],
                                  samples[i],
                                  samples_n[i],
                                  gt_mask[i, mask[i]])
            loss += (losses_weights*losses).sum(-1)
            loss_t += (losses_weights*losses).cpu().detach().numpy()
        loss_items.append(loss_t / sdfs.shape[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0)
        optimizer.step()
    return loss_items


if __name__ == '__main__':
    device = 'cuda'
    parser = argparse.ArgumentParser(
        description='CNN training on ABC dataset')
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Data loading
    with open(cfg['data']['names'], 'r') as f:
        train_set_names = [e[:-1] for e in f.readlines()]

    dataset = ABCDataset_multiple(
        cfg['data']['hdf5'], train_set_names, cfg['data']['grid_n'], True, subsample=cfg['training']['sample_fac'])
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['training']['batch_size'], shuffle=True, pin_memory=True)
    print('Loaded {} training shapes'.format(len(dataset)))

    if 'input' in cfg['training']:
        print('Loading pre-trained model {}'.format(cfg['training']['input']))
        M = torch.load('models/'+cfg['training']['input'])
    else:
        M = CNN_3d_multiple_split(
            cfg['data']['grid_n'], K=cfg['training']['K'], ef_dim=128).to(device)
        # M = CNN_3d_multiple_quadrics_batchnorm(
        #     cfg['data']['grid_n'], K=cfg['training']['K']).to(device)
    M.train()
    global optimizer
    optimizer = torch.optim.AdamW(M.parameters(), cfg['training']['lr'], weight_decay=cfg['training']['wd'], betas=(
        cfg['training']['beta1'], cfg['training']['beta2']), amsgrad=cfg['training']['amsgrad'])

    losses_weights = torch.tensor(
        cfg['training']['losses_weights'], dtype=torch.float32, device=device)

    L = []
    for i in tqdm(range(cfg['training']['epochs'])):
        L += train(M, optimizer, train_loader,
                   losses_weights)
        if i % 1 == 0:
            plt.clf()
            losses = np.row_stack(L)
            labels = ['chamfer', 'vstars', 'loss_normals',
                      'loss_quadrics', 'loss_bools']
            for i in range(5):
                plt.plot(losses[:, i], label=labels[i])
            plt.legend()
            plt.yscale('log')
            plt.legend()
            plt.savefig('loss.png', bbox_inches='tight')
            torch.save(M, 'model.pt')
