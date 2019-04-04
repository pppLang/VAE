import os
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from vae import VAE, KLLoss
from train import train_epoch, test


def main():
    train_dataset = datasets.MNIST(root='C:/Users/user/Documents/InterestingAttempt/VAE/mnist_data/', train=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    test_dataset = datasets.MNIST(root='C:/Users/user/Documents/InterestingAttempt/VAE/mnist_data/', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

    criterion = nn.MSELoss()
    criterion2 = KLLoss()

    epoch_num = 80
    lr = 1e-3
    weight_decay = 1e-5
    lamda = 0.1
    latent_num = 2

    outf = r'C:\Users\user\Documents\InterestingAttempt\VAE\logs\linear2_{}_{}_{}_{}_{}'.format(latent_num, lr, lamda, weight_decay, epoch_num)
    if not os.path.exists(outf):
        os.makedirs(outf)
    
    model = VAE(28*28, latent_num).cuda()
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, betas=(0.9, 0.999))
    writer = SummaryWriter(outf)
    for epoch in range(epoch_num):
        current_lr = lr / 2**int(epoch/20)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        train_epoch(model, optimizer, train_loader, criterion, epoch, writer=writer, criterion2=criterion2, lamda=lamda)
        test(model, test_loader, criterion, epoch, writer=writer, criterion2=criterion2)
        if (epoch+1)%10==0:
            torch.save(model.state_dict(), os.path.join(outf, 'model_{}.pth'.format(epoch)))
    writer.close()
    torch.save(model.state_dict(), os.path.join(outf, 'model.pth'))


if __name__=="__main__":
    main()