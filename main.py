import os
import torch
from torch import nn
from torchvision import datasets, transforms
from vae import VAE, KLLoss


def main():
    train_dataset = datasets.MNIST(root='../mnist_data/', train=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    test_dataset = datasets.MNIST(root='../mnist_data/', train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

    criterion = nn.MSELoss()
    criterion2 = KLLoss()

    epoch_num = 50
    