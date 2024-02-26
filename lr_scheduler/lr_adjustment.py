import math
import types
import numpy as np
import torch
from bisect import bisect_right
from torch.optim.optimizer import Optimizer


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    def display_lr(self):
        for _lr in self.get_lr():
            print(f"\n🔉 当前epoch的学习率：{format(_lr, '.6f')}")



class LambdaLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        super(LambdaLR, self).__init__(optimizer, last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        lr = [base_lr * lmbda(self.last_epoch) for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]
        return lr


class CyclicLR(_LRScheduler):

    def __init__(self, optimizer, base_lr, max_lr, step_size, gamma=0.99, mode='triangular', last_epoch=-1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.mode = mode
        assert mode in ['triangular', 'triangular2', 'exp_range']
        super(CyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lr = []
        for base_lr in self.base_lrs:
            cycle = np.floor(1 + self.last_epoch / (2 * self.step_size))
            x = np.abs(float(self.last_epoch) / self.step_size - 2 * cycle + 1)
            if self.mode == 'triangular':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
            elif self.mode == 'triangular2':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
            elif self.mode == 'exp_range':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * (self.gamma ** (
                    self.last_epoch))
            new_lr.append(lr)
        return new_lr


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
            self,
            optimizer,
            warmup_start_lr,
            warmup_epochs,
            max_epochs,
            warmup='exp',
            cos_eta=0,
            last_epoch=-1
    ):
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup = warmup
        self.cos_eta = cos_eta
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # exp lr warmup
            if self.warmup == 'exp':
                lr_group = [
                    self.warmup_start_lr * (
                            pow(lr / self.warmup_start_lr, 1. / self.warmup_epochs)
                            ** self.last_epoch
                    )
                    for lr in self.base_lrs
                ]
            # linear warmup
            elif self.warmup == 'linear':
                warmup_factor = self.last_epoch / self.warmup_epochs
                lr_group = [
                    self.warmup_start_lr
                    + (lr - self.warmup_start_lr) * warmup_factor
                    for lr in self.base_lrs
                ]
        else:
            cos_last_epoch = self.last_epoch - self.warmup_epochs
            cos_epochs = self.max_epochs - self.warmup_epochs
            cos_factor = (1 + math.cos(math.pi * cos_last_epoch / cos_epochs)) / 2.
            lr_group = [
                self.cos_eta + (lr - self.cos_eta) * cos_factor
                for lr in self.base_lrs
            ]
        return lr_group


class WarmupMultiStepLR(_LRScheduler):
    def __init__(
            self,
            optimizer,
            warmup_start_lr,
            warmup_epochs,
            milestones,
            gamma=0.1,
            warmup='exp',
            last_epoch=-1
    ):
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.gamma = gamma
        self.warmup = warmup
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # exp lr warmup
            if self.warmup == 'exp':
                lr_group = [
                    self.warmup_start_lr * (
                            pow(lr / self.warmup_start_lr, 1. / self.warmup_epochs)
                            ** self.last_epoch
                    )
                    for lr in self.base_lrs
                ]
            # linear warmup
            elif self.warmup == 'linear':
                warmup_factor = self.last_epoch / self.warmup_epochs
                lr_group = [
                    self.warmup_start_lr
                    + (lr - self.warmup_start_lr) * warmup_factor
                    for lr in self.base_lrs
                ]
        else:
            lr_group = [
                lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for lr in self.base_lrs
            ]
        return lr_group


class WarmupCyclicLR(_LRScheduler):
    def __init__(
            self,
            optimizer,
            warmup_start_lr,
            warmup_epochs,
            max_epochs,
            cycle_len,
            cycle_mult,
            lr_decay=1,
            warmup='exp',
            cos_eta=0,
            last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult
        self.lr_decay = lr_decay
        self.cos_eta = cos_eta
        self.warmup = warmup
        self.lr_decay_factor = 1
        self.n_cycles = 0
        self.curr_cycle_len = cycle_len
        self.cycle_past_all_epoch = 0
        super(WarmupCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # exp lr warmup
            if self.warmup == 'exp':
                lr_group = [
                    self.warmup_start_lr * (
                            pow(lr / self.warmup_start_lr, 1. / self.warmup_epochs)
                            ** self.last_epoch
                    )
                    for lr in self.base_lrs
                ]
            # linear warmup
            elif self.warmup == 'linear':
                warmup_factor = self.last_epoch / self.warmup_epochs
                lr_group = [
                    self.warmup_start_lr
                    + (lr - self.warmup_start_lr) * warmup_factor
                    for lr in self.base_lrs
                ]
        else:
            cycle_epoch = (
                    self.last_epoch - self.warmup_epochs - self.cycle_past_all_epoch
            )
            if cycle_epoch > self.curr_cycle_len:
                self.cycle_past_all_epoch += self.curr_cycle_len
                self.curr_cycle_len *= self.cycle_mult
                cycle_epoch = 0
                self.lr_decay_factor *= self.lr_decay
            cos_factor = 0.5 * (
                    1 + math.cos(math.pi * cycle_epoch / self.curr_cycle_len)
            )
            lr_group = [
                self.cos_eta + (lr * self.lr_decay_factor - self.cos_eta)
                * cos_factor
                for lr in self.base_lrs
            ]
        return lr_group


class WarmupLR:
    def __init__(self, optimizer, num_warm) -> None:
        self.optimizer = optimizer
        self.num_warm = num_warm
        self.lr = [group['lr'] for group in self.optimizer.param_groups]
        self.num_step = 0

    def __compute(self, lr) -> float:
        return lr * min(self.num_step ** (-0.5), self.num_step * self.num_warm ** (-1.5))

    def step(self) -> None:
        self.num_step += 1
        lr = [self.__compute(lr) for lr in self.lr]
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr[i]


# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# scheduler = WarmupLR(optimizer=optimizer, num_warm=10)
# lr_history = scheduler_lr(optimizer, scheduler)
#
# optimizer2 = torch.optim.SGD(model.parameters(), lr=1e-3)
# scheduler2 = WarmupLR(optimizer=optimizer2, num_warm=20)
# lr_history2 = scheduler_lr(optimizer2, scheduler2)
