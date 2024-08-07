from typing import List
import torch
import torch.nn as nn
from typing import Optional


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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        return self.focal_loss_impl(pred, target)


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


class ClassificationLoss(FocalLoss):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma=0  # type:(int,float)
    ):
        super().__init__(alpha, gamma)


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
        :param outputs: (N,C,H,W)
        :param targets: (N,H,W),(N,1,H,W)
        """
        assert len(outputs.shape) == 4, "Output tensor must be 4D (N, C, H, W)"

        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        else:
            raise ValueError("Target tensors.shape should be (N,H,W),(N,1,H,W)")

        targets = targets.long()

        bce = 0
        dice = 0
        iou = 0

        if self.weights[0] != 0:
            bce = self.bce_loss(outputs, targets) * self.weights[0]

        if self.weights[1] != 0:
            dice = self.dice_loss(outputs, targets) * self.weights[1]

        if self.weights[2] != 0:
            iou = self.iou_loss(outputs, targets) * self.weights[2]

        total_loss = bce + dice + iou
        # total_loss = (bce * self.weights[0] + dice * self.weights[1] + iou * self.weights[2]) / sum(self.weights)
        return total_loss
