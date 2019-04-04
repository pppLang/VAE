import torch
import matplotlib.pyplot as plt


def train_epoch(model, optimizer, train_loader, criterion, epoch, writer=None, criterion2=None, lamda=1):
    model.train()
    num = len(train_loader)
    for i, (data, _) in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        data = data.cuda()
        data = data.reshape([data.shape[0], data.shape[1]*data.shape[2]*data.shape[2]])
        result, mean, std = model(data)
        rec_loss = criterion(result, data)
        KL_loss = criterion2(mean, std)
        loss = rec_loss + lamda*KL_loss
        loss.backward()
        optimizer.step()
        if i%10==0:
            print('epoch {}, [{}/{}], loss {}'.format(epoch, i, num, loss))
            if writer is not None:
                writer.add_scalar('loss', loss.item(), epoch*num + i)
                writer.add_scalar('rec_loss', rec_loss.item(), epoch*num + i)
                writer.add_scalar('KL_loss', KL_loss.item(), epoch*num + i)


def test(model, test_loader, criterion, epoch, writer=None, criterion2=None):
    model.eval()
    test_loss = 0
    KL_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.cuda()
            data = data.reshape([data.shape[0], data.shape[1]*data.shape[2]*data.shape[2]])
            result, mean, std = model(data)
            test_loss += criterion(result, data).item()
            KL_loss += criterion2(mean, std).item()
    num = len(test_loader)
    test_loss, KL_loss = test_loss/num, KL_loss/num
    print('epoch {}, rec loss {}, KL loss {}'.format(epoch, test_loss, KL_loss))
    if writer is not None:
        writer.add_scalar('test_rec_loss', test_loss, epoch)
        writer.add_scalar('test_KL_loss', KL_loss, epoch)

        show_result = result[0:3,:]
        show_result = show_result.cpu().numpy().reshape([3,28,28])
        fig = plt.figure()
        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.imshow(show_result[i,:,:])
        writer.add_figure('show_result', fig, epoch)

        show_input = data[0:3,:]
        print(show_input.shape)
        show_input = show_input.cpu().numpy().reshape([3,28,28])
        fig = plt.figure()
        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.imshow(show_input[i,:,:])
        writer.add_figure('show_input', fig, epoch)