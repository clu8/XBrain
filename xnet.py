import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class XNet(nn.Module):
    def __init__(self):
        super(XNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, dilation=2), # 1016
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(4), # 254
            nn.Conv2d(32, 64, kernel_size=5, padding=3), # 256
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(4), # 64
            nn.Conv2d(64, 64, kernel_size=5), # 60
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2), # 30
            Flatten(),
            nn.Linear(30 * 30 * 64, 1),
            nn.Sigmoid()
        )

        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, X):
        return self.net(X)

    def train_step(self, X, y):
        preds = self(X)
        weights = y.data + 1 # weight positives 2:1
        loss_fn = nn.BCELoss(weight=weights)
        loss = loss_fn(preds, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
