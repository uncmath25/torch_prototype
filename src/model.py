import torch
import torch.nn as nn


class TorchPrototypeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(28, 28, 100)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(28*28, 10)

    def forward(self, x):
        # x = self.conv(x)
        # x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
