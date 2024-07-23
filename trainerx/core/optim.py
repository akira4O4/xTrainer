from contextlib import contextmanager
from typing import Dict, Optional, List

import torch
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler

__all__ = [
    'OptimWrapper',
    'AMPOptimWrapper'
]


class OptimWrapper:
    def __init__(self, optimizer: Optimizer):
        assert isinstance(optimizer, Optimizer), ('optimizer must be a `torch.optim.Optimizer` instance, but got '
                                                  f'{type(optimizer)}')
        self.optimizer = optimizer
        self._update_count = 0

    def zero_grad(self, **kwargs) -> None:
        self.optimizer.zero_grad(**kwargs)

    @property
    def param_groups(self) -> List[dict]:
        return self.optimizer.param_groups

    @property
    def lrs(self) -> list:
        # optimizer.param_groups[0]:"[‘params’, ‘lr’, ‘betas’, ‘eps’, ‘weight_decay’, ‘amsgrad’, ‘maximize’]
        lr = [group['lr'] for group in self.param_groups]
        return lr

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    def __call__(
        self,
        loss: torch.Tensor,
        loss_kwargs: Optional[Dict] = None,
        step_kwargs: Optional[Dict] = None,
        zero_kwargs: Optional[Dict] = None
    ) -> None:
        loss_kwargs = {} if loss_kwargs is None else loss_kwargs
        step_kwargs = {} if step_kwargs is None else step_kwargs
        zero_kwargs = {} if zero_kwargs is None else zero_kwargs

        loss.backward(**loss_kwargs)
        self.optimizer.step(**step_kwargs)
        self.zero_grad(**zero_kwargs)

    @contextmanager
    def context(self):
        yield


class AMPOptimWrapper(OptimWrapper):
    def __init__(
        self,
        loss_scale: Optional[str] = 'dynamic',
        **kwargs
    ):
        super().__init__(**kwargs)
        self._scale_update_param = None
        #  If loss_scale is a string, it must be 'dynamic', then dynamic loss scaling will be used.
        if loss_scale == 'dynamic':
            self.grad_scaler = GradScaler()

            # Static loss scaling
        elif isinstance(loss_scale, float):
            self._scale_update_param = loss_scale
            self.grad_scaler = GradScaler(init_scale=loss_scale)

        # More specific configuration.
        elif isinstance(loss_scale, dict):
            self.grad_scaler = GradScaler(**loss_scale)

        else:
            raise TypeError(f'loss_scale must be of type float, dict, or dynamic", but got {loss_scale}')

    def state_dict(self) -> dict:
        state_dict = self.optimizer.state_dict()
        state_dict['loss_scaler'] = self.grad_scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        if 'loss_scaler' in state_dict:
            self.grad_scaler.load_state_dict(state_dict.pop('loss_scaler'))
        self.optimizer.load_state_dict(state_dict)

    def __call__(
        self,
        loss: torch.Tensor,
        loss_kwargs: Optional[Dict] = None,
        step_kwargs: Optional[Dict] = None,
        zero_kwargs: Optional[Dict] = None
    ) -> None:

        loss_kwargs = {} if loss_kwargs is None else loss_kwargs
        step_kwargs = {} if step_kwargs is None else step_kwargs
        zero_kwargs = {} if zero_kwargs is None else zero_kwargs

        self.grad_scaler.scale(loss).backward(**loss_kwargs)
        self.grad_scaler.step(self.optimizer, **step_kwargs)
        self.grad_scaler.update()
        self.zero_grad(**zero_kwargs)

    @contextmanager
    def context(self):
        from torch.cuda.amp import autocast
        with autocast():
            yield
