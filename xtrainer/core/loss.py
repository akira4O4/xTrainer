from typing import Dict, List
import torch
import torch.nn as nn
from typing import Optional

'''
pt=softmax(x)
FocalLoss(pt)=-alpha(1-pt)^gamma*log(pt)
alpha调和不同类别中的数据不平衡，可以输入一个数字如0.25（第一类数据的比重为0.25，其余类别为1-0.25），也可以直接输入一个list，顺序为各类的权重（数据量小的给大权重）
gamma负责降低简单样本的损失值, hardcase的预测的分数低，easycase的概率高，（1-p）**gamma可以将easycase的loss进一步降低，反之亦然
'''


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma=0  # type:(int,float)
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # [nc,]

    # classification prediction.shape=(bs,nc) target.shape=(bs,)
    # segmentation prediction.shape=(bs,nc,h,w) target.shape=(bs,h,w)
    def focal_loss_impl(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

        # pred (bs, nc)  or  (bs, nc, h, w, ...)
        # target (bs, )  or  (bs, h, w, ...)
        bs, nc = pred.shape[:2]
        if pred.dim() > 2:
            pred = pred.reshape(bs, nc, -1)  # (bs, nc, h, w) => (bs, nc, h*w)
            pred = pred.transpose(1, 2)  # (bs, nc, h*w) => (bs, h*w, nc)
            pred = pred.reshape(-1, nc)  # (bs, h*w, nc) => (bs*h*w, nc)   set N = bs*h*w

        target = target.reshape(-1)  # (N, )

        log_p = torch.log_softmax(pred, dim=-1)  # (N, nc)
        log_p = log_p.gather(dim=1, index=target[:, None])  # log_p.shape=(N,1)
        log_p = log_p.squeeze()  # (N,1)->(N,)
        p = torch.exp(log_p)  # (N,)  e^(log(x))=x

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


# Dice=(2x∣A∩B∣)/(∣A∣+∣B∣)

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super(DiceLoss, self).__init__()
        self.smooth = smooth

        # smooth = 1e-6：适用于极小的目标区域，通常能有效防止数值稳定性问题。
        # smooth = 1.0：适用于一般情况，能够处理小区域并且不引入过大的偏差。

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算多类别Dice损失。

        :param outputs: 模型的输出张量，形状为[batch_size, num_classes, height, width]。
        :param targets: 真实标签张量，形状为[batch_size, height, width]。
        :return: 计算得到的Dice损失值。
        """
        num_classes = outputs.size(1)
        dice = 0

        outputs = torch.softmax(outputs, dim=1)
        for i in range(num_classes):
            output_i = outputs[:, i, :, :]
            target_i = (targets == i).float()
            output_i = output_i.contiguous().view(-1)
            target_i = target_i.contiguous().view(-1)
            intersection = (output_i * target_i).sum()
            total = (output_i + target_i).sum()
            dice += (2. * intersection + self.smooth) / (total + self.smooth)

        return 1 - dice / num_classes


class IoULoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算多类别IoU损失。
        :param outputs: 模型的输出张量，形状为[batch_size, num_classes, height, width]。
        :param targets: 真实标签张量，形状为[batch_size, height, width]。
        :return: 计算得到的IoU损失值。
        """
        num_classes = outputs.size(1)
        iou = 0

        outputs = torch.softmax(outputs, dim=1)
        for i in range(num_classes):
            output_i = outputs[:, i, :, :]
            target_i = (targets == i).float()
            intersection = (output_i * target_i).sum()
            union = (output_i + target_i - output_i * target_i).sum()
            iou += (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou / num_classes


class SegmentationLoss(nn.Module):
    def __init__(self, weights: List[float] = None) -> None:
        super(SegmentationLoss, self).__init__()

        if weights is None:
            weights = [1.0, 1.0, 1.0]

        if len(weights) != 3:
            raise ValueError("weights列表的长度必须为3")

        self.weights = weights
        self.bce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算加权总损失。

        :param outputs: 模型的输出张量，形状为[batch_size, num_classes, height, width]。
        :param targets: 真实标签张量，形状为[batch_size, height, width] 或 [batch_size, 1, height, width]。
        :return: 计算得到的加权总损失值。
        """

        if targets.dim() == 4:
            targets = targets.squeeze(1)  # 调整目标张量的形状为 [batch_size, height, width]
        targets = targets.long()

        bce: torch.Tensor = self.bce_loss(outputs, targets)
        dice: torch.Tensor = self.dice_loss(outputs, targets)
        iou: torch.Tensor = self.iou_loss(outputs, targets)
        # print(f'bce:{bce.cpu().item()},dice{dice.cpu().item()},iou:{iou.cpu().item()}')
        total_loss = bce * self.weights[0] + dice * self.weights[1] + iou * self.weights[2]
        # total_loss = (bce * self.weights[0] + dice * self.weights[1] + iou * self.weights[2]) / sum(self.weights)
        return total_loss


if __name__ == "__main__":
    # criterion: nn.Module = DiceLoss()
    #
    # inputs: torch.Tensor = torch.randn(1, 1, 256, 256, requires_grad=True)  # Example prediction
    # targets: torch.Tensor = torch.randint(0, 2, (1, 1, 256, 256)).float()  # Example ground truth
    #
    # loss: torch.Tensor = criterion(inputs, targets)
    #
    # print("Dice Loss:", loss.item())

    import numpy as np
    from torch.nn import functional as F

    # bs, nc, h, w = 32, 4, 100, 200
    # pred = np.random.randn(bs, nc, h, w)
    # target = np.random.randint(0, nc, size=(bs, h, w))
    # target = torch.tensor(target, dtype=torch.long)

    bs, nc = 32, 4
    pred = np.random.randn(bs, nc)
    target = np.random.randint(0, nc, size=(bs,))
    target = torch.tensor(target, dtype=torch.long)
    pred_logit1 = torch.tensor(pred, dtype=torch.float, requires_grad=True)
    pred_logit2 = torch.tensor(pred, dtype=torch.float, requires_grad=True)

    print(f'pred_logit1.dim(): {pred_logit1.dim()}')
    print(f'pred_logit2.dim(): {pred_logit2.dim()}')
    print(f'target.dim(): {target.dim()}')

    alpha = np.abs(np.random.randn(nc))
    alpha = torch.tensor(alpha, dtype=torch.float)

    loss1 = FocalLoss(gamma=0.0, alpha=alpha)(pred_logit1, target)
    loss1.backward()

    loss2 = F.cross_entropy(pred_logit2, target, weight=alpha)
    loss2.backward()

    print(loss1)
    print(loss2)
    print(pred_logit1.grad[1, 2])
    print(pred_logit2.grad[1, 2])
