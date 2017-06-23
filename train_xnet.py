import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from xnet import XNet
import config


BATCH_SIZE = 16

def get_data_loader(X, y):
    return DataLoader(
        dataset=TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1
    )

def check_accuracy(xnet, loader):
    num_correct = 0
    num_samples = 0
    xnet.eval()

    for X, y in loader:
        X_var = Variable(X.cuda(), volatile=True)

        scores = xnet(X_var)
        preds = scores.data.cpu().numpy().flatten() > 0.5
        num_correct += (preds == y.numpy()).sum()
        num_samples += len(preds)

    acc = num_correct / num_samples
    tqdm.write('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

def train(xnet, loader_train, loader_test, num_epochs=25, print_every=10):
    print('Training')

    for epoch in range(num_epochs):
        tqdm.write('Starting epoch {} / {}'.format(epoch + 1, num_epochs))

        xnet.train()

        for i, (X, y) in tqdm(enumerate(loader_train)):
            X_var = Variable(X.cuda(), requires_grad=False)
            y_var = Variable(y.type(torch.FloatTensor).cuda(), requires_grad=False)

            loss = xnet.train_step(X_var, y_var)

            if i % print_every == 0:
                tqdm.write('i = {}, loss = {:.4}'.format(i + 1, loss.data[0]))

        check_accuracy(xnet, loader_train)
        check_accuracy(xnet, loader_test)

if __name__ == '__main__':
    with open(config.PROCESSED_PATH, 'rb') as f:
        X_train, y_train, X_test, y_test = pickle.load(f)

    loader_train = get_data_loader(X_train, y_train)
    loader_test = get_data_loader(X_test, y_test)

    print('Data loaders ready')

    xnet = XNet().cuda()
    train(xnet, loader_train, loader_test)
