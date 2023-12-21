import torch
from torch import nn
import mesh_tools as mt
from PoNQ import QuadricBaseNN


class resnet_block(nn.Module):
    def __init__(self, ef_dim):
        super(resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim,
                                1, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim,
                                1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.conv_1(input)
        output = nn.functional.leaky_relu(
            output, negative_slope=0.01, inplace=True)
        output = self.conv_2(output)
        output = output+input
        output = nn.functional.leaky_relu(
            output, negative_slope=0.01, inplace=True)
        return output


class SDF_decoder(nn.Module):
    def __init__(self, out_features, scale=1, K=1, ef_dim=64, decoder_layers=3, grid_n=33) -> None:
        super().__init__()
        self.grid_n = grid_n
        self.ef_dim = ef_dim
        self.scale = scale
        self.out_features = out_features
        self.K = K
        self.decoder = []
        for i in range(decoder_layers):
            self.decoder.append(resnet_block(self.ef_dim))
        self.decoder.append(
            nn.Conv3d(self.ef_dim, self.out_features*self.K, 1, bias=True))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x = self.decoder(x).view(x.shape[0], self.out_features*self.K,
                                 (self.grid_n-1)**3).permute((0, 2, 1))
        return x.view(
            x.shape[0], (self.grid_n-1)**3, self.K, self.out_features)/self.scale



class CNN_3d_multiple_split(QuadricBaseNN):
    def __init__(self, grid_n=33, encoder_layers=5, decoder_layers=3, K=4, ef_dim=128, device="cuda"):
        super().__init__()
        self.grid_n = grid_n
        self.grid = torch.tensor(mt.mesh_grid(
            self.grid_n-1, True)*(grid_n-1)/grid_n, dtype=torch.float32).to(device)
        self.ef_dim = ef_dim
        self.K = K  # number of points
        # convolutions
        self.encoder = []
        self.encoder.append(
            nn.Conv3d(1, self.ef_dim, 2, stride=1, bias=True)
        )
        self.encoder.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        for _ in range(encoder_layers):
            self.encoder.append(
                nn.Conv3d(self.ef_dim, self.ef_dim, 3,
                          stride=1, padding=1, bias=True)
            )
            self.encoder.append(nn.LeakyReLU(
                negative_slope=0.01, inplace=True))
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder_points = SDF_decoder(
            3, scale=grid_n, K=K, ef_dim=self.ef_dim, decoder_layers=decoder_layers, grid_n=grid_n)
        self.decoder_vstars = SDF_decoder(
            3, scale=grid_n, K=K, ef_dim=self.ef_dim, decoder_layers=decoder_layers, grid_n=grid_n)
        self.decoder_As = SDF_decoder(
            6, scale=1, K=K, ef_dim=self.ef_dim, decoder_layers=decoder_layers, grid_n=grid_n)
        self.decoder_mean_normals = SDF_decoder(
            3, scale=1, K=K, ef_dim=self.ef_dim, decoder_layers=decoder_layers, grid_n=grid_n)
        self.decoder_bools = SDF_decoder(
            1, scale=1, K=1, ef_dim=self.ef_dim, decoder_layers=decoder_layers, grid_n=grid_n)

    def change_grid_size(self, grid_n):
        self.grid_n = grid_n
        self.grid = torch.tensor(mt.mesh_grid(
            self.grid_n-1, True)*(grid_n-1)/grid_n, dtype=torch.float32).to(self.grid.device)
        self.decoder_points.grid_n = grid_n
        self.decoder_points.scale = grid_n
        self.decoder_vstars.grid_n = grid_n
        self.decoder_vstars.scale = grid_n
        self.decoder_As.grid_n = grid_n
        self.decoder_mean_normals.grid_n = grid_n
        self.decoder_bools.grid_n = grid_n

    def forward(self, x):
        x = self.encoder(x)
        predicted_points = self.decoder_points(x) + self.grid[None, :, None, :]
        predicted_vstars = self.decoder_vstars(x) + predicted_points.detach()
        predicted_As = self.get_As(self.decoder_As(x))
        predicted_mean_normals = self.decoder_mean_normals(x)
        predicted_bool = torch.sigmoid(
            self.decoder_bools(x).squeeze(-1).squeeze(-1))
        return predicted_points, predicted_vstars, predicted_mean_normals, predicted_As, predicted_bool