from typing import Optional
import math
from torch import optim
from torch.optim.lr_scheduler import LambdaLR


class LRSchedulerWrapper:
    def __init__(
        self,
        optimizer: optim.Optimizer,
        last_epoch: Optional[int] = -1,
        lrf: Optional[float] = 0.01,
        epochs: Optional[int] = 100,
        cos_lr: Optional[bool] = False,
    ):
        if cos_lr:
            lr_lambda = self._cos_lr_lambda(1, lrf, epochs)
        else:
            lr_lambda = self._linear_lr_lambda(epochs, lrf)

        self.scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

    @staticmethod
    def _easy_lr_lambda():
        return lambda x: 1 / (x / 4 + 1),

    @staticmethod
    def _cos_lr_lambda(y1: float = 0.0, y2: float = 1.0, steps: int = 100):
        return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1

    @staticmethod
    def _linear_lr_lambda(epochs: int, lrf: float):
        return lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf  # linear

    def func(self):
        return self.scheduler

    def __call__(self, epoch: Optional[int] = None):
        self.scheduler.step(epoch)


if __name__ == '__main__':
    import torch
    import torch.nn as nn


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)

        def forward(self, x=1):
            out = self.conv(x)
            return out


    net = Net()

    optimizer = torch.optim.SGD([{'params': net.parameters(), 'initial_lr': 1}], lr=1)
    scheduler = LRSchedulerWrapper(optimizer)

    for i in range(9):
        print("lr of epoch", i, "=>", scheduler.func().get_last_lr())
        optimizer.step()
        scheduler()
