import torch
import mesh_tools as mt
from tqdm import tqdm 
from PoNQ import PoNQ


def sumpool(tens):
    return tens[..., ::2, ::2, ::2]+tens[..., 1::2, ::2, ::2]+tens[..., ::2, 1::2, ::2]+tens[..., ::2, ::2, 1::2]+tens[..., 1::2, ::2, 1::2]+tens[..., ::2, 1::2, 1::2]+tens[..., 1::2, 1::2, 1::2]+tens[..., 1::2, 1::2, ::2]


def CNN_to_PoNQ(model, sdfs, grid_n, mask, subd=0, device="cuda"):
    # output PoNQ from network, subd=0: PoNQ subd=1: PoNQ-lite, subd>1: PoNQ-lite subdivided
    with torch.no_grad():
        model.change_grid_size(grid_n)
        _, predicted_vstars, predicted_mean_normals, predicted_quadrics, predicted_bool = model(
            sdfs.to(device)*(grid_n-1)/32)
        final_mask = (predicted_bool > .5)*mask.to(device)
        if subd == 0:
            predicted_quadrics = predicted_quadrics[0,
                                                    final_mask[0]].view(-1, 3, 3)
            predicted_vstars, predicted_mean_normals = predicted_vstars[0, final_mask[0]].view(
                -1, 3), predicted_mean_normals[0, final_mask[0]].view(-1, 3)
            bs = (predicted_quadrics @
                  predicted_vstars[..., None]).squeeze(-1)
            cs = (bs*predicted_vstars).sum(-1)
            predicted_Q = torch.zeros(
                (len(bs), 4, 4), dtype=torch.float32, device=bs.device)
            predicted_Q[:, :3, :3] = predicted_quadrics
            predicted_Q[:, 3, :3] = -bs
            predicted_Q[:, :3, 3] = -bs
            predicted_Q[:, 3, 3] = cs

        else:
            # PoNQ lite
            predicted_bool = (predicted_bool > .5)*1.*mask.to(device)
            bs = (predicted_quadrics @ predicted_vstars[..., None]).squeeze(-1)
            cs = (bs*predicted_vstars).sum(-1)
            bs.shape, cs.shape
            predicted_Q = torch.zeros(
                (bs.shape[0], bs.shape[1], 4, 4, 4), dtype=torch.float32, device=bs.device)
            predicted_Q[:, :, :, :3, :3] = predicted_quadrics
            predicted_Q[:, :, :, 3, :3] = -bs
            predicted_Q[:, :, :, :3, 3] = -bs
            predicted_Q[:, :, :, 3, 3] = cs
            predicted_Q = predicted_Q.mean(-3)
            # only serve as references
            predicted_vstars = predicted_vstars.mean(-2)
            predicted_mean_normals = predicted_mean_normals.mean(-2)

            predicted_vstars = predicted_vstars.view(
                -1, grid_n-1, grid_n-1, grid_n-1, 3).permute((0, 4, 1, 2, 3))
            predicted_mean_normals = predicted_mean_normals.view(
                -1, grid_n-1, grid_n-1, grid_n-1, 3).permute((0, 4, 1, 2, 3))
            predicted_Q = predicted_Q.view(-1, grid_n-1,
                                           grid_n-1, grid_n-1, 16).permute((0, 4, 1, 2, 3))
            predicted_bool = predicted_bool.view(-1,
                                                 grid_n-1, grid_n-1, grid_n-1)

            for _ in range(subd-1):
                # Further subdivisions
                predicted_vstars = sumpool(
                    predicted_vstars*(predicted_bool[:,  None] > 0))
                predicted_mean_normals = sumpool(
                    predicted_mean_normals*(predicted_bool[:,  None] > 0))
                predicted_Q = sumpool(
                    predicted_Q*(predicted_bool[:,  None] > 0))
                predicted_bool = sumpool(predicted_bool)

            predicted_vstars /= predicted_bool[:, None]
            predicted_mean_normals /= predicted_bool[:, None]
            predicted_Q /= predicted_bool[:, None]
            predicted_bool = predicted_bool > 0
            predicted_vstars = predicted_vstars.permute((0, 2, 3, 4, 1))[
                predicted_bool > 0]
            predicted_mean_normals = predicted_mean_normals.permute((0, 2, 3, 4, 1))[
                predicted_bool > 0]
            predicted_Q = predicted_Q.permute((0, 2, 3, 4, 1))[
                predicted_bool > 0].view(-1, 4, 4)
        V = PoNQ(predicted_vstars.cpu(
        ).detach().numpy(), device)
        V.mean_normals = predicted_mean_normals
        V.quadrics = V.quadric_matrix_to_vector(predicted_Q)
        V.non_void[:] = 1
        V.get_vstars()
        return V


def CNN_to_PoNQ_large(model, sdfs, grid_n, mask, subd=0, kernel_size=65, device="cuda"):
    stride = (kernel_size-1)//2
    with torch.no_grad():
        patches = sdfs.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)
        patches = patches.reshape(1, -1, kernel_size, kernel_size, kernel_size).permute(1, 0, 2, 3, 4)

        patch_mask = mask.reshape(1, 1, grid_n-1, grid_n-1,grid_n-1)
        patch_mask = patch_mask.unfold(2, kernel_size-1, stride).unfold(3, kernel_size-1, stride).unfold(4, kernel_size-1, stride)
        patch_mask = patch_mask.reshape(1, -1,  kernel_size-1,  kernel_size-1,  kernel_size-1).permute(1, 0, 2, 3, 4)
        patch_mask = patch_mask.reshape(-1, (kernel_size-1)**3)

        patch_grid = torch.tensor(mt.mesh_grid(grid_n-1, True)*(grid_n-1)/grid_n, dtype=torch.float32)
        patch_grid = patch_grid.reshape(grid_n-1, grid_n-1, grid_n-1, 3).permute((3, 0, 1, 2))
        patch_grid = patch_grid.unfold(1, kernel_size-1, stride).unfold(2, kernel_size-1, stride).unfold(3, kernel_size-1, stride)
        patch_grid = patch_grid.reshape(3, -1, (kernel_size-1), (kernel_size-1), (kernel_size-1))
        patch_grid = patch_grid.reshape(3, -1, (kernel_size-1)**3).permute((1, 2, 0))
        i=0
        model.change_grid_size(kernel_size)
        model.decoder_vstars.scale = grid_n
        model.decoder_points.scale = grid_n
        all_vstars = []
        all_mean_normals = []
        all_quadrics = []
        for i in tqdm(range(len(patches))):
            model.grid = patch_grid[i].to(device)
            _, predicted_vstars, predicted_mean_normals, predicted_quadrics, predicted_bool = model(patches[None, i]*(grid_n-1)/32)
            start = stride//2
            end = kernel_size-1-stride//2
            final_mask = (predicted_bool*patch_mask[i]).reshape(kernel_size-1, kernel_size-1, kernel_size-1)[start:end, start:end, start:end]>.5
            predicted_vstars = predicted_vstars.reshape(kernel_size-1, kernel_size-1, kernel_size-1, 4, 3)[start:end, start:end, start:end][final_mask]
            predicted_mean_normals = predicted_mean_normals.reshape(kernel_size-1, kernel_size-1, kernel_size-1, 4, 3)[start:end, start:end, start:end][final_mask]
            predicted_quadrics = predicted_quadrics.reshape(kernel_size-1, kernel_size-1, kernel_size-1, 4, 3, 3)[start:end, start:end, start:end][final_mask]
            
            all_vstars.append(predicted_vstars)
            all_mean_normals.append(predicted_mean_normals)
            all_quadrics.append(predicted_quadrics)
            torch.cuda.empty_cache()
        predicted_vstars = torch.cat(all_vstars)
        predicted_mean_normals = torch.cat(all_mean_normals)
        predicted_quadrics = torch.cat(all_quadrics)
        bs = (predicted_quadrics @ predicted_vstars[..., None]).squeeze(-1)
        cs = (bs*predicted_vstars).sum(-1)
        predicted_Q = torch.zeros(
            (bs.shape[0], 4, 4, 4), dtype=torch.float32, device=bs.device)
        predicted_Q[:, :, :3, :3] = predicted_quadrics
        predicted_Q[:, :, 3, :3] = -bs
        predicted_Q[:, :, :3, 3] = -bs
        predicted_Q[:, :, 3, 3] = cs
        if subd>1:
            return "WARNING: further subdivision not allowed in large grid unfolding"
        elif subd==1: #lite
            predicted_Q = predicted_Q.mean(-3)
            predicted_vstars = predicted_vstars.mean(-2)
            predicted_mean_normals = predicted_mean_normals.mean(-2)
        else:
            predicted_Q = predicted_Q.view(-1, 4, 4)
            predicted_vstars = predicted_vstars.view(-1, 3)
            predicted_mean_normals = predicted_mean_normals.view(-1, 3)

        V = PoNQ(predicted_vstars.cpu(
        ).detach().numpy(), device)
        V.mean_normals = predicted_mean_normals
        V.quadrics = V.quadric_matrix_to_vector(predicted_Q)
        V.non_void[:] = 1
        V.get_vstars()
        return V