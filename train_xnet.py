import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from xnet import XNet
import config
from logging_utils import Logger


BATCH_SIZE = 16

def get_data_loader(X, y):
    return DataLoader(
        dataset=TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1
    )

def evaluate(xnet, loader):
    num_samples = 0
    num_tp, num_tn, num_fp, num_fn = 0, 0, 0, 0

    xnet.eval()

    for X, y in loader:
        X_var = Variable(X.cuda(), volatile=True)

        scores = xnet(X_var)
        preds = scores.data.cpu().numpy().flatten() > 0.5
        answers = y.numpy()

        num_samples += len(preds)
        num_tp += ((preds == answers) & (answers == 1)).sum()
        num_tn += ((preds == answers) & (answers == 0)).sum()
        num_fp += ((preds != answers) & (answers == 0)).sum()
        num_fn += ((preds != answers) & (answers == 1)).sum()

    acc = (num_tp + num_tn) / num_samples
    precision = num_tp / (num_tp + num_fp)
    recall = num_tp / (num_tp + num_fn)
    f1 = 2 * precision * recall / (precision + recall)

    logger.log_print('n = {} | acc = {:.3f} | precision = {:.3f} | recall = {:.3f} | F1 = {:.3f}'.format(
        num_samples,
        acc,
        precision,
        recall,
        f1
    ))
    return (acc, precision, recall, f1)

def train(xnet, loader_train, loader_test, num_epochs=25, print_every=10, save_model=True):
    logger.log_print('Training')
    best_test_precision = 0

    for epoch in range(num_epochs):
        logger.log_print('Starting epoch {} / {}'.format(epoch + 1, num_epochs))

        xnet.train()

        for i, (X, y) in tqdm(enumerate(loader_train)):
            X_var = Variable(X.cuda(), requires_grad=False)
            y_var = Variable(y.type(torch.FloatTensor).cuda(), requires_grad=False)

            loss = xnet.train_step(X_var, y_var)

            if i % print_every == 0:
                logger.log_print('i = {}, loss = {:.4}'.format(i + 1, loss.data[0]))

        if epoch % 5 == 4:
            logger.log_print('Evaluating on training set')
            evaluate(xnet, loader_train)

        logger.log_print('Evaluating on test set')
        acc, precision, recall, f1 = evaluate(xnet, loader_test)

        if precision > best_test_precision and recall > 0.97 and save_model:
            logger.log_print('Saving new best model')
            best_test_precision = precision
            torch.save(xnet.state_dict(), config.MODEL_PATH)

if __name__ == '__main__':
    logger = Logger('log.txt')

    with open(config.PROCESSED_PATH, 'rb') as f:
        X_train, y_train, X_test, y_test = pickle.load(f)

    loader_train = get_data_loader(X_train, y_train)
    loader_test = get_data_loader(X_test, y_test)

    logger.log_print('Data loaders ready')

    xnet = XNet().cuda()
    train(xnet, loader_train, loader_test)

    logger.close()
