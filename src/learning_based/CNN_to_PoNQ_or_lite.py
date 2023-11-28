import torch
from src.utils.PoNQ import PoNQ


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
