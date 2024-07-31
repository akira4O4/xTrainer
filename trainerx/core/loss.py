import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import shutil

'''
pt=softmax(x)
FocalLoss(pt)=-alpha(1-pt)^gamma*log(pt)
alpha调和不同类别中的数据不平衡，可以输入一个数字如0.25（第一类数据的比重为0.25，其余类别为1-0.25），也可以直接输入一个list，顺序为各类的权重（数据量小的给大权重）
gamma负责降低简单样本的损失值, hardcase的预测的分数低，easycase的概率高，（1-p）**gamma可以将easycase的loss进一步降低，反之亦然
'''


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):

        super(MultiClassFocalLossWithAlpha, self).__init__()
        if alpha is None:
            alpha = [0.2, 0.3, 0.5]
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1)  # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  # 对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma=0  # type:(int,float)
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # [nc,]

    # classification prediction.shape=(bs,nc) target.shape=(bs,1)
    # segmentation prediction.shape=(bs,1,h,w) target.shape=(bs,1,h,w)
    def focal_loss_impl(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred (bs, nc)  or  (bs, nc, X1, X2, ...)
        # target (bs, )  or  (bs, X1, X2, ...)
        bs, nc = pred.shape[:2]  # batch size and number of categories
        if pred.dim() > 2:
            # e.g. pred.shape is (bs, nc, X1, X2]
            pred = pred.reshape(bs, nc, -1)  # (bs, nc, X1, X2) => (bs, nc, X1*X2)
            pred = pred.transpose(1, 2)  # (bs, nc, X1*X2) => (bs, X1*X2, nc)
            pred = pred.reshape(-1, nc)  # (bs, X1*X2, nc) => (bs*X1*X2, nc)   set N = bs*X1*X2

        target = target.reshape(-1)  # (N, )

        log_p = torch.log_softmax(pred, dim=-1)  # (N, nc)
        log_p = log_p.gather(1, target[:, None]).squeeze()  # (N,)
        p = torch.exp(log_p)  # (N,)

        if self.alpha is None:
            self.alpha = torch.ones((nc,), dtype=torch.float, device=pred.device)

        self.alpha = self.alpha.gather(0, target)  # [N,]

        loss = -1 * self.alpha * torch.pow(1 - p, self.gamma) * log_p
        return loss.sum() / self.alpha.sum()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal_loss_impl(pred, target)


class ClassificationLoss(FocalLoss):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma=0  # type:(int,float)
    ):
        super().__init__(alpha, gamma)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to inputs if they are logits
        inputs = torch.sigmoid(inputs)

        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        # Calculate Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (total + self.smooth)

        # Calculate Dice loss
        dice_loss = 1 - dice_coeff

        return dice_loss


class SegmentationLoss:
    def __init__(self):
        ...


if __name__ == "__main__":
    criterion: nn.Module = DiceLoss()

    inputs: torch.Tensor = torch.randn(1, 1, 256, 256, requires_grad=True)  # Example prediction
    targets: torch.Tensor = torch.randint(0, 2, (1, 1, 256, 256)).float()  # Example ground truth

    loss: torch.Tensor = criterion(inputs, targets)

    print("Dice Loss:", loss.item())

    # import numpy as np
    #
    # bs, nc, X1, X2 = 32, 4, 100, 200
    # pred = np.random.randn(bs, nc, X1, X2)
    # pred_logit1 = torch.tensor(pred, dtype=torch.float, requires_grad=True)
    # pred_logit2 = torch.tensor(pred, dtype=torch.float, requires_grad=True)
    #
    # target = np.random.randint(0, nc, size=(bs, X1, X2))
    # target = torch.tensor(target, dtype=torch.long)
    #
    # alpha = np.abs(np.random.randn(nc))
    # alpha = torch.tensor(alpha, dtype=torch.float)
    #
    # loss1 = FocalLoss(gamma=0.0, alpha=alpha)(pred_logit1, target)
    # loss1.backward()
    #
    # loss2 = F.cross_entropy(pred_logit2, target, weight=alpha)
    # loss2.backward()
    #
    # print(loss1)
    # print(loss2)
    # print(pred_logit1.grad[1, 2, 3, 4])
    # print(pred_logit2.grad[1, 2, 3, 4])
