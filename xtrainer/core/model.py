import os
from typing import Optional, Any, Dict

import torch
import torchvision
from loguru import logger

from xtrainer import network
from xtrainer.utils.common import error_exit

__all__ = ['Model']


class Model:
    def __init__(
        self,
        model_name: str,
        num_classes: Optional[int] = 0,
        mask_classes: Optional[int] = 0,
        pretrained: Optional[bool] = False,
        weight: Optional[str] = None,
        device: Optional[int] = 0,  # -1==cpu
        strict: Optional[bool] = False,
        map_location: Optional[str] = 'cpu',
    ):
        self._is_gpu = False
        self._model_name = model_name
        self._pretrained = pretrained
        self._num_classes = num_classes
        self._mask_classes = mask_classes
        self._weight = weight
        self._checkpoint: Dict[str, Any] = {}
        self._strict = strict
        self._device = torch.device('cpu')
        self._map_location = map_location

        self.set_device(device)

        self._net = None

    @property
    def training(self) -> bool:
        return self._net.training

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def state_dict(self) -> dict:
        return self._net.state_dict()

    @property
    def device(self) -> torch.device:
        return self._device

    def set_device(self, idx: int) -> None:
        if not torch.cuda.is_available() or idx == -1:
            self._device = torch.device('cpu')
            self._is_gpu = False
        else:
            self._device = torch.device(f'cuda:{idx}')
            self._is_gpu = True
        logger.info(f'Setting model to {self._device}')

    @property
    def is_gpu(self) -> bool:
        return self._is_gpu

    @property
    def checkpoint(self) -> Dict[str, Any]:
        return self._checkpoint

    @property
    def parameters(self):
        return self._net.parameters()

    def __call__(self, images: torch.Tensor) -> Any:

        if self.training:
            return self._net(images)

        else:
            with torch.no_grad():
                return self._net(images)

    def train(self) -> None:
        self._net.train()

    def eval(self) -> None:
        self._net.eval()

    def to_device(self) -> None:
        self._net.to(self._device)

    def set_weight(self, path: str) -> None:
        self._weight = path

    def set_net(self, val) -> None:
        self._net = val

    def init(self) -> None:
        self.build_model()

        self.load_weight()

        self.to_device()

    def build_model(self) -> None:
        net = network.__dict__.get(self.model_name, None)

        if net is None:
            from torchvision import models
            net = models.__dict__.get(self.model_name, None)

        if net is None:
            logger.error(f'Don`t get the model:{self.model_name}.')
            error_exit()

        net_args = {
            'num_classes': self._num_classes,
        }

        if self._mask_classes != 0:
            net_args['mask_classes'] = self._mask_classes

        tv_version = torchvision.__version__.split('.')  # 0.13.1=['0','13','1']

        if tv_version[1] < '13':
            net_args['pretrained'] = self._pretrained
        else:
            net_args['weights'] = self._pretrained
            # net_args['weights'] = None

        self._net = net(**net_args)
        logger.info('Build Model Done.')

    def load_weight(self) -> None:

        if (self._weight is None) or (os.path.exists(self._weight) is False):
            logger.warning('Weight is not found.')
            return

        self._checkpoint = torch.load(self._weight, map_location=self._map_location)
        weight_state_dict = self.checkpoint.get('state_dict', None)

        if weight_state_dict is None:
            logger.error('Weight.state_dict is not found.')
            return

        model_state_dict = self._net.state_dict()

        total_item = 0
        loading_item = 0

        for (net_key, net_value), (weight_key, weight_value) in zip(model_state_dict.items(),
                                                                    weight_state_dict.items()):
            total_item += 1

            if net_value.shape == weight_value.shape:
                model_state_dict[net_key] = weight_value
                loading_item += 1

        self._net.load_state_dict(model_state_dict, strict=self._strict)

        logger.info(f'Loading :{self._weight}.')
        logger.info(f'Loading [{loading_item}/{total_item}] item to model.')


if __name__ == '__main__':
    model = Model(
        'resnet18',
        num_classes=2,
        pretrained=False,
        device=0
    )
    model.init()
    model.train()
    print(model.training)
    model.eval()
    print(model.training)
