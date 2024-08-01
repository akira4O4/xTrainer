import torch
import torch.nn as nn
import random

__all__ = ['Hswish', 'Hsigmoid', 'Identity', 'SEModule', 'SPPF']


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3.) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return self.relu6(x + 3.) / 6.


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            # nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            # nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False),

            Hsigmoid(True)
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.avg_pool(x).view(n, c)
        out = self.fc(out).view(n, c, 1, 1)
        return x * out.expand_as(x)


class ECAModule(nn.Module):
    def __init__(self, in_channels, ):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        b = 1
        gamma = 2
        import math
        k = int(abs((math.log(in_channels, 2) + b) / gamma))
        k = k if k % 2 else k + 1
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=(k - 1) // 2, bias=False),
            Hsigmoid(True)
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.avg_pool(x).view(n, 1, c)
        out = self.conv(out).view(n, c, 1, 1)
        return x * out


class Identity(nn.Module):
    def __init__(self, in_channels):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        # self.cv1 = nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0)
        self.cv1 = nn.Sequential(nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(c_),
                                 nn.ReLU(inplace=True))

        # self.cv2 = nn.Conv2d(c_*4, c2, kernel_size=1, stride=1, padding=0)
        self.cv2 = nn.Sequential(nn.Conv2d(c_ * 4, c2, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(c2),
                                 nn.ReLU(inplace=True))
        self.p = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)  # 先通过CBL进行通道数的减半
        y1 = self.p(x)
        y2 = self.p(y1)
        y3 = self.p(y2)
        # 上述两次最大池化
        # return self.cv2(torch.cat([x, y1, y2, self.p(x)], 1))
        return self.cv2(torch.cat([x, y1, y2, y3], 1))
        # 将原来的x,一次池化后的y1,两次池化后的y2,3次池化的self.m(y2)先进行拼接，然后再CBL


class MultiSampleDropout(nn.Module):
    def __init__(self, conv, dim=1, dropout_num=4, p=0.5, average=False):
        super(MultiSampleDropout, self).__init__()
        self.dropout_num = dropout_num
        self.conv = conv
        self.average = average

        assert dim <= 2
        if dim == 1:
            self.dropout_ops = nn.ModuleList(
                [nn.Dropout2d(p) for _ in range(self.dropout_num)]
            )
        elif dim == 2:
            self.dropout_ops = nn.ModuleList(
                [nn.Dropout2d(p) for _ in range(self.dropout_num)]
            )

    def forward(self, x):
        logits = None
        if self.training:
            for i, dropout_op in enumerate(self.dropout_ops):
                if i == 0:
                    out = dropout_op(x)
                    logits = self.conv(out) * random.random()
                else:
                    temp_out = dropout_op(x)
                    temp_logits = self.conv(temp_out) * random.random()
                    logits = logits + temp_logits

        else:
            x = self.dropout_ops[0](x)
            logits = self.conv(x) * self.dropout_num * 0.5

        if self.average:
            logits = logits / self.dropout_num

        return logits
