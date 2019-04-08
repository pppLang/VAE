import torch
import matplotlib.pyplot as plt
import numpy as np


def train_epoch(model,
                optimizer,
                train_loader,
                criterion,
                epoch,
                writer=None,
                criterion2=None,
                lamda=1):
    model.train()
    num = len(train_loader)
    for i, (data, _) in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        data = data.cuda()
        data = data.reshape(
            [data.shape[0], data.shape[1] * data.shape[2] * data.shape[2]])
        result, mean, std, z = model(data)
        rec_loss = criterion(result, data)#, size_average=False)
        KL_loss = criterion2(mean, std)
        loss = rec_loss + lamda * KL_loss
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('epoch {}, [{}/{}], loss {}, rec loss {}, KL loss {}'.format(epoch, i, num, loss, rec_loss, KL_loss))
            if writer is not None:
                writer.add_scalar('loss', loss.item(), epoch * num + i)
                writer.add_scalar('rec_loss', rec_loss.item(), epoch * num + i)
                writer.add_scalar('KL_loss', KL_loss.item(), epoch * num + i)
                writer.add_scalar('z_mean', z.mean(), epoch * num + i)
                writer.add_scalar('z_min', z.min(), epoch * num + i)
                writer.add_scalar('z_max', z.max(), epoch * num + i)


def test(model, test_loader, criterion, epoch, writer=None, criterion2=None):
    model.eval()
    test_loss = 0
    KL_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.cuda()
            data = data.reshape(
                [data.shape[0], data.shape[1] * data.shape[2] * data.shape[2]])
            result, mean, std, z = model(data)
            test_loss += criterion(result, data).item()
            KL_loss += criterion2(mean, std).item()
    num = len(test_loader)
    test_loss, KL_loss = test_loss / num, KL_loss / num
    print('epoch {}, rec loss {}, KL loss {}'.format(epoch, test_loss,
                                                     KL_loss))
    if writer is not None:
        writer.add_scalar('test_rec_loss', test_loss, epoch)
        writer.add_scalar('test_KL_loss', KL_loss, epoch)
        writer.add_scalar('test_mean', mean.mean().item(), epoch)
        writer.add_scalar('test_std', std.mean().item(), epoch)
        writer.add_scalar('test_z', z.mean().item(), epoch)

        """ show_result = result[0:3, :]
        show_result = show_result.cpu().numpy().reshape([3, 28, 28])
        fig = plt.figure()
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(show_result[i, :, :])
        writer.add_figure('show_result', fig, epoch)

        show_input = data[0:3, :]
        print(show_input.shape)
        show_input = show_input.cpu().numpy().reshape([3, 28, 28])
        fig = plt.figure()
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(show_input[i, :, :])
        writer.add_figure('show_input', fig, epoch) """

        rec_raw_num = 4
        rec_column_num = 10
        rec_show = np.zeros([rec_raw_num*2*28, rec_column_num*28])
        for i in range(rec_raw_num):
            for j in range(rec_column_num):
                index = i*rec_column_num + j
                gt = data[index, :].cpu().numpy().reshape([28 ,28])
                rec = result[index, :].cpu().numpy().reshape([28 ,28])
                rec_show[i*2*28:(i*2+1)*28, j*28:(j+1)*28] = gt
                rec_show[(i*2+1)*28:(i+1)*2*28, j*28:(j+1)*28] = rec
        fig = plt.figure()
        plt.imshow(rec_show)
        writer.add_figure('show_rec_img', fig, epoch)

        muti_img = plot_muti_img(model, z.min().item(), z.max().item(), 15)
        print(muti_img.shape)
        fig = plt.figure()
        plt.imshow(muti_img)
        writer.add_figure('show_muti_img', fig, epoch)


def plot_muti_img(model, min_value, max_value, num):
    x1_linspace = np.linspace(min_value, max_value, num)
    x2_linspace = np.linspace(min_value, max_value, num)
    muti_img = np.zeros([num*28, num*28])
    for i, x1 in enumerate(x1_linspace):
        for j, x2 in enumerate(x2_linspace):
            x = np.array([[x1, x2]])
            x = torch.Tensor(x).cuda()
            with torch.no_grad():
                y = model.decoder(x)
            img = np.squeeze(y.cpu().numpy()).reshape([28, 28])
            muti_img[i*28:(i+1)*28, j*28:(j+1)*28] = img
    return muti_img


if __name__ == "__main__":
    x = np.array([[0, 1]])
    print(x.shape)