import torch
import torch.nn.functional as F
import torch.nn as nn
import loss
import sys
from typing import List, Union, Optional
from register import LOSS
from loguru import logger

Tensor = torch.Tensor
List_Tensor = List[torch.Tensor]

__all__ = [
    'CrossEntropyLoss',
    'PeriodLoss',
    'DiceLoss',
]


class LossBaseCalc:
    def __init__(
            self,
            model_output,
            targets: Tensor,
            **kwargs
    ):
        self.model_output = model_output
        self.targets = targets

    def forward(self):
        pass


@LOSS.registered
class CrossEntropyLoss(LossBaseCalc):
    def __init__(
            self,
            model_output: Union[Tensor, List_Tensor],
            targets: Tensor,
            **kwargs
    ):
        super().__init__(model_output, targets, **kwargs)
        # export.shape:
        # Tensor
        # [Tensor1,Tensor2,...]

        self.model_output = model_output
        self.targets = targets
        self.loss_func = loss.CrossEntropyLoss(**kwargs)

    def forward(self):
        _loss: float = 0.0

        if isinstance(self.model_output, list):  # [Tensor1,Tensor2,...] or [cls_output,seg_output]
            self.model_output = self.model_output[0]
            for n_l in range(len(self.model_output)):
                _loss += self.loss_func(self.model_output[n_l], self.targets)
            _loss = _loss / len(self.model_output)

        elif isinstance(self.model_output, Tensor):
            _loss = self.loss_func(self.model_output, self.targets)
        return _loss


@LOSS.registered
class PeriodLoss(LossBaseCalc):
    def __init__(
            self,
            model_output: List_Tensor,
            targets: Tensor,
            weight: list,
            device: torch.device,
            **kwargs
    ):
        # export.shape:
        # [Tensor1,Tensor2,...]

        super().__init__(model_output, targets, **kwargs)
        self.model_output = model_output
        self.targets = targets
        kwargs.update({'period_weights': torch.tensor(weight).float().to(device)})
        self.loss_func = loss.PeriodLoss(**kwargs)

    def forward(self) -> float:
        if isinstance(self.model_output, list):  # [Tensor1,Tensor2,...] or [cls_output,seg_output]
            if isinstance(self.model_output[0], list):  # [Tensor1,Tensor2,...]
                self.model_output = self.model_output[1]

        _loss = self.loss_func([self.model_output[0], self.model_output[2]], self.targets)
        _loss += 0.5 * self.loss_func([self.model_output[1], self.model_output[3]], self.targets)
        return _loss


@LOSS.registered
class DiceLoss(LossBaseCalc):
    def __init__(
            self,
            model_output: List_Tensor,
            targets: Tensor,
            layer_weights: Optional[List] = None,
            **kwargs
    ):
        super().__init__(model_output, targets, **kwargs)
        self.model_output = model_output
        self.targets = targets
        self.loss_func = loss.DiceLoss(**kwargs)
        if layer_weights is None:
            self.layer_weights = [1, 1, 0.5, 0.5]
        else:
            self.layer_weights = layer_weights

    def forward(self) -> float:
        _loss = 0.0
        if isinstance(self.model_output, list):
            if isinstance(self.model_output[1], list):
                self.model_output = self.model_output[1]

            for n_l in range(len(self.model_output)):
                target_t = self.targets
                if (self.model_output[n_l].shape[2] != target_t.shape[2]) and (
                        self.model_output[n_l].shape[2] / target_t.shape[2] == self.model_output[n_l].shape[3] /
                        target_t.shape[3]):
                    target_t = F.interpolate(target_t,
                                             (self.model_output[n_l].shape[2], self.model_output[n_l].shape[3]))
                # target_t = torch.tensor(target_t, dtype=torch.int64)
                target_t = torch.as_tensor(target_t, dtype=torch.int64)
                _loss += self.layer_weights[n_l] * self.loss_func(self.model_output[n_l], target_t)
        return _loss
