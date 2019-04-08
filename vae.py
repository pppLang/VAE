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
        fea = self.linear1(x)
        mean = self.relu(self.linear_mean(fea))
        variance = self.relu(self.linear_variance(fea))
        return mean, variance


class Decoder(nn.Module):
    def __init__(self, latent_num, mid_features, out_features):
        super(Decoder, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(latent_num, mid_features),
            nn.Linear(mid_features, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linears(x)


class VAE(nn.Module):
    def __init__(self, features, latent_num):
        super(VAE, self).__init__()
        self.encoder = Encoder(features, latent_num)
        self.decoder = Decoder(latent_num, features)

    def sample(self, mean, std):
        epspilon = torch.normal(mean=torch.zeros_like(mean), std=torch.ones_like(mean))
        return mean + std*epspilon

    def forward(self, x):
        mean, std = self.encoder(x)
        z = self.sample(mean, std)
        x_ = self.decoder(z)
        return x_, mean, std


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mean, std):
        std2 = std * std
        mean2 = mean * mean
        return 0.5*torch.mean(mean2 + std2 - torch.log(std2) - 1)


if __name__ == "__main__":
    vae = VAE(28*28, 2)
    x = torch.rand(64, 28*28)
    print(x.shape)