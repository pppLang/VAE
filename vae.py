import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_features, mid_features, latent_num):
        super(Encoder, self).__init__()
        self.latent_num = latent_num
        self.linear1 = nn.Linear(in_features, mid_features)
        self.linear_mean = nn.Linear(mid_features, latent_num)
        self.linear_variance = nn.Linear(mid_features, latent_num)
        self.relu = nn.ReLU()

    def forward(self, x):
        fea = self.relu(self.linear1(x))
        mean = self.linear_mean(fea)
        variance = self.linear_variance(fea)
        return mean, variance


class Decoder(nn.Module):
    def __init__(self, latent_num, mid_features, out_features):
        super(Decoder, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(latent_num, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linears(x)


class VAE(nn.Module):
    def __init__(self, features, mid_features, latent_num):
        super(VAE, self).__init__()
        self.encoder = Encoder(features, mid_features, latent_num)
        self.decoder = Decoder(latent_num, mid_features, features)

    def sample(self, z_mean, z_log_std):
        epspilon = torch.normal(mean=torch.zeros_like(z_mean), std=torch.ones_like(z_mean))
        std = torch.exp(z_log_std/2)
        # epspilon = torch.randn_like(std)
        return z_mean + std*epspilon

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sample(z_mean, z_log_var)
        x_ = self.decoder(z)
        return x_, z_mean, z_log_var, z


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, z_mean, z_log_var):
        loss = - 0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return loss


if __name__ == "__main__":
    vae = VAE(28*28, 256, 2)
    x = torch.rand(64, 28*28)
    print(x.shape)