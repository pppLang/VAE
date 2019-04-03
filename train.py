import torch


def train_epoch(model, optimizer, train_loader, criterion, epoch, writer=None, criterion2=None):
    model.train()
    num = len(train_loader)
    for i, data in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        data = data.cuda()
        result = model(data)
        rec_loss = criterion(result, data)
        KL_loss = criterion2(result)
        loss = rec_loss + KL_loss
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
        for i, data in enumerate(test_loader):
            data = data.cuda()
            result = model(data)
            test_loss += criterion(result, data).item()
            KL_loss += criterion2(result).item()
    num = len(test_loader.dataset)
    test_loss, KL_loss = test_loss/num, KL_loss/num
    print('epoch {}, rec loss {}, KL loss {}'.format(epoch, test_loss, KL_loss))
    if writer is not None:
        writer.add_scalar('rec_loss', test_loss, epoch)
        writer.add_scalar('KL_loss', KL_loss, epoch)