from contextlib import contextmanager
from typing import Dict, Optional, List

import torch
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler

__all__ = [
    'OptimWrapper',
    'AmpOptimWrapper'
]


class OptimWrapper:
    def __init__(self, optimizer: Optimizer):
        assert isinstance(optimizer, Optimizer), (
            'optimizer must be a `torch.optim.Optimizer` instance, but got '
            f'{type(optimizer)}')
        self.optimizer = optimizer
        self._update_count = 0

    # @property
    # def optimizer(self):
    #     return self.optimizer
    #
    # @optimizer.setter
    # def optimizer(self, value):
    #     self._optimizer = value

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        loss.backward(**kwargs)
        self._update_count += 1

    def step(self, **kwargs) -> None:
        self.optimizer.step(**kwargs)

    def zero_grad(self, **kwargs) -> None:
        self.optimizer.zero_grad(**kwargs)

    @property
    def param_groups(self) -> List[dict]:
        return self.optimizer.param_groups

    @property
    def lr(self) -> list:
        lr = [group['lr'] for group in self.param_groups]
        return lr

    def show_lr(self) -> None:
        for _lr in self.lr:
            print(f"\n🔉 当前epoch的学习率：{format(_lr, '.8f')}")

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    def update_params(
            self,
            loss: torch.Tensor,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None
    ) -> None:
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        self.backward(loss)
        self.step(**step_kwargs)
        self.zero_grad(**zero_kwargs)


class AmpOptimWrapper(OptimWrapper):
    def __init__(
            self,
            loss_scale: str = 'dynamic',
            **kwargs
    ):
        super().__init__(**kwargs)
        self._scale_update_param = None
        if loss_scale == 'dynamic':
            #  If loss_scale is a string, it must be 'dynamic', then dynamic
            #  loss scaling will be used.
            self.loss_scaler = GradScaler()
        elif isinstance(loss_scale, float):
            # Static loss scaling
            self._scale_update_param = loss_scale
            self.loss_scaler = GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            # More specific configuration.
            self.loss_scaler = GradScaler(**loss_scale)
        else:
            raise TypeError('loss_scale must be of type float, dict, or '
                            f'"dynamic", but got {loss_scale}')

    def backward(self, loss: torch.Tensor, **kwargs):
        self.loss_scaler.scale(loss).backward(**kwargs)
        self._update_count += 1

    def step_update(self, **kwargs):
        self.loss_scaler.step(self.optimizer, **kwargs)
        self.loss_scaler.update(self._scale_update_param)

    def update_params(
            self,
            loss: torch.Tensor,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None
    ) -> None:
        if zero_kwargs is None:
            zero_kwargs = {}
        self.zero_grad(**zero_kwargs)
        self.loss_scaler.scale(loss).backward()
        self.loss_scaler.step(self.optimizer)
        self.loss_scaler.update()

    def state_dict(self) -> dict:
        state_dict = self.optimizer.state_dict()
        state_dict['loss_scaler'] = self.loss_scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        if 'loss_scaler' in state_dict:
            self.loss_scaler.load_state_dict(state_dict.pop('loss_scaler'))
        self.optimizer.load_state_dict(state_dict)

    @contextmanager
    def optim_context(self):
        from torch.cuda.amp import autocast
        with autocast():
            yield
