from typing import List, Union, Optional

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainerx.loss.period_loss import PeriodLoss
from trainerx.loss.dice_loss import DiceLoss

__all__ = [
    'BaseLossForward',
    'CrossEntropyLossForward',
    'DiceLossForward',
    'PeriodLossForward',
    'LOSS_FORWARD_TABLE'
]


class BaseLossForward:
    def __init__(
        self,
        model_output: torch.Tensor,
        targets: torch.Tensor,
    ):
        self._model_output = model_output
        self._targets = targets
        self.loss = None  # loss instance
        self._loss_val = -1.0
        self._loss_name = ''

    @property
    def loss_val(self) -> float:
        return round(self._loss_val, 6)

    @property
    def loss_name(self) -> str:
        return self._loss_name

    def set_model_output(self, val) -> None:
        self._model_output = val

    def set_targets(self, val) -> None:
        self._targets = val

    def build(self) -> None:
        ...

    def run(self):
        ...


class CrossEntropyLossForward(BaseLossForward):
    def __init__(
        self,
        model_output: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ):
        super().__init__(model_output, targets)
        self.kwargs = kwargs
        self._loss_name = 'CrossEntropyLoss'

    def build(self):
        self.loss = nn.CrossEntropyLoss(**self.kwargs)
        logger.success('Build CrossEntropyLoss Done.')

    def forward(self):
        loss = self.loss(self._model_output, self._targets)
    def run(self):
        _loss: float = 0.0

        if isinstance(self._model_output, list):  # [Tensor1,Tensor2,...] or [cls_output,seg_output]
            self._model_output = self._model_output[0]
            for n_l in range(len(self._model_output)):
                _loss += self.loss(self._model_output[n_l], self._targets)
            _loss = _loss / len(self._model_output)

        elif isinstance(self._model_output, torch.Tensor):
            _loss = self.loss(self._model_output, self._targets)
        return _loss


class PeriodLossForward(BaseLossForward):
    def __init__(
        self,
        model_output: Optional[List[torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
        weight: Optional[list] = None,
        device: Optional[torch.device] = torch.device('cuda:0'),
        **kwargs
    ):
        super().__init__(model_output, targets)
        self.kwargs = kwargs
        self._weight = weight
        self._device = device
        self._loss_name = 'PeriodLoss'

    @property
    def weight(self):
        return self._weight

    def set_weight(self, val) -> None:
        self._weight = val

    @property
    def device(self) -> torch.device:
        return self._device

    def set_device(self, val: torch.device) -> None:
        self._device = val

    def build(self) -> None:
        self.kwargs.update({
            'period_weights': torch.tensor(self.weight).float().to(self.device)
        })
        self.loss = PeriodLoss(**self.kwargs)
        logger.success('Build PeriodLoss Done.')

    def run(self) -> float:
        if isinstance(self._model_output, list):  # [Tensor1,Tensor2,...] or [cls_output,seg_output]
            if isinstance(self._model_output[0], list):  # [Tensor1,Tensor2,...]
                self._model_output = self._model_output[1]

        _loss = self.loss([self._model_output[0], self._model_output[2]], self._targets)
        _loss += 0.5 * self.loss([self._model_output[1], self._model_output[3]], self._targets)
        return _loss


class DiceLossForward(BaseLossForward):
    def __init__(
        self,
        model_output: Optional[List[torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
        layer_weights: Optional[List] = None,
        **kwargs
    ):
        super().__init__(model_output, targets)
        self._layer_weights = layer_weights
        self.kwargs = kwargs
        self._loss_name = 'DiceLoss'

    @property
    def layer_weights(self):
        return self._layer_weights

    def set_layer_weights(self, val) -> None:
        self._layer_weights = val

    def build(self) -> None:
        self.loss = DiceLoss(**self.kwargs)

        if self.layer_weights is None:
            self._layer_weights = [1, 1, 0.5, 0.5]
        else:
            self._layer_weights = self.layer_weights

        logger.success('Build DiceLoss Done.')

    def run(self) -> float:
        _loss = 0.0
        if isinstance(self._model_output, list):
            if isinstance(self._model_output[1], list):
                self._model_output = self._model_output[1]

            for n_l in range(len(self._model_output)):
                target_t = self._targets
                if (self._model_output[n_l].shape[2] != target_t.shape[2]) and (
                    self._model_output[n_l].shape[2] / target_t.shape[2] == self._model_output[n_l].shape[3] /
                    target_t.shape[3]):
                    target_t = F.interpolate(target_t,
                                             (self._model_output[n_l].shape[2], self._model_output[n_l].shape[3]))
                # target_t = torch.tensor(target_t, dtype=torch.int64)
                target_t = torch.as_tensor(target_t, dtype=torch.int64)
                _loss += self.layer_weights[n_l] * self.loss(self._model_output[n_l], target_t)
        return _loss


LOSS_FORWARD_TABLE = {
    'CrossEntropyLoss': CrossEntropyLossForward,
    'PeriodLoss': PeriodLossForward,
    'DiceLoss': DiceLossForward,
}

if __name__ == '__main__':
    ...
