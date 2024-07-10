import math
from typing import Optional
import numpy as np
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
softmax(x)=exp(x)/sum(exp(X))
CE=-log(softmax(x))
FL=-alpha_x*(1-softmax(x))^gamma*log(softmax(x))
"""


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: int = 2,
        weight: Optional[torch.Tensor] = None
    ) -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # weight==FL alpha

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)

        # note:exp(ln(x))=x
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class _FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


if __name__ == '__main__':
    # input = torch.randn(3, requires_grad=True)
    # target = torch.empty(3).random_(2)
    data = 2
    data1 = math.log(data)
    print(data1)
    print(math.exp(data1))
