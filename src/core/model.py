import os
import warnings
from typing import Optional
from collections import OrderedDict

import torch
from loguru import logger

from src import network
from src.utils.util import get_time

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
        strict: Optional[bool] = True,
        map_location: Optional[str] = 'cpu',
    ):
        self._is_gpu = False
        self._is_cpu = True
        self._model_name = model_name
        self._pretrained = pretrained
        self._num_classes = num_classes
        self._mask_classes = mask_classes
        self._weight = weight
        self._strict = strict
        self._device = torch.device('cpu')
        self._map_location = map_location

        if device != -1 and torch.cuda.is_available():
            self._device = torch.device(f'cuda:{device}')
            self._is_gpu = True
            self._is_cpu = False
        else:
            self._device = torch.device('cpu')
            self._is_gpu = False
            self._is_cpu = True

        self._net = None

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self._net(images)

    @property
    def training(self) -> bool:
        return self._net.training

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def mask_classes(self) -> int:
        return self._mask_classes

    @property
    def state_dict(self) -> dict:
        return self._net.state_dict()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_cpu(self) -> bool:
        return self._is_cpu

    @property
    def is_gpu(self) -> bool:
        return self._is_gpu

    def set_gpu(self, gpu: int) -> None:
        self._device = f'cuda:{gpu}'
        logger.info(f'Change device to {self._device}')

    @property
    def parameters(self):
        return self._net.parameters()

    def train(self) -> None:
        self._net.train()

    def eval(self) -> None:
        self._net.eval()

    def move_to_device(self) -> None:
        self._net.to(self._device)
        logger.info(f'Move model.to: {self._device}.')

    def set_weight(self, path: str) -> None:
        self._weight = path

    def init(self) -> None:
        self.build_model()
        self.load_weight()
        self.move_to_device()

    def build_model(self) -> None:
        net = network.__dict__.get(self.model_name)
        if net is None:
            logger.error(f'Don`t get the model:{self.model_name}.')
            exit()

        self._net = net(
            num_classes=self._num_classes,
            mask_classes=self._mask_classes,
            pretrained=self._pretrained,
            input_channels=3,
        )

    def load_weight(self) -> None:

        weight = torch.load(self._weight, map_location=self._map_location)
        weight_state_dict = weight.get('state_dict')

        model_state_dict = self._net.state_dict()

        if weight_state_dict is None:
            logger.error(f'weight do not found the state_dict.')
            return

        total_item = 0
        loading_item = 0

        for k, v in weight_state_dict.items():
            total_item += 1

            if 'module' in k:  # DDP model
                k = k[7:]

            if k in model_state_dict.keys():
                if v.shape == model_state_dict[k].shape:
                    model_state_dict[k] = v
                    loading_item += 1

        self._net.load_state_dict(model_state_dict, strict=self._strict)

        logger.info(f'Loading :{self._weight}.')
        logger.info(f'Loading [{loading_item}/{total_item}] item to model.')

    def save_checkpoint(
        self,
        save_path: str,
        **kwargs,
    ) -> None:

        save_dict = {"state_dict": self.state_dict}

        if kwargs != {}:
            save_dict.update(kwargs)

        epoch = kwargs.get('epoch', 0)

        model_save_path = os.path.join(save_path, f"epoch{epoch}.pth")
        torch.save(save_dict, model_save_path)
        logger.success(f'👍 Save weight to: {save_path}.')

    # def remove_ddp_key(self) -> dict:
    #     warnings.warn("", DeprecationWarning)
    #     del_count = 0
    #     del_ddp_key_flag = False
    #     new_state_dict = OrderedDict()
    #     for k, v in self.state_dict.items():
    #         if 'module' in k:
    #             # k = k[7:]
    #             k = k.replace('module.', '')
    #             del_count += 1
    #             del_ddp_key_flag = True
    #
    #         new_state_dict[k] = v
    #
    #     if del_ddp_key_flag:
    #         logger.info(f'Delete num of key(module) of DDP model: {del_count} item.')
    #
    #     return new_state_dict

    # def ddp_mode(self, sync_bn: bool = False) -> None:
    #
    #     warnings.warn("", DeprecationWarning)
    #     if self._gpu is None or self._gpu == -1:
    #         logger.warning(f'Your GPU:{self._gpu}.')
    #         return
    #
    #     if self.net is None:
    #         logger.error(f'Model is not init.')
    #         return
    #
    #     if sync_bn:
    #         self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
    #         logger.info(f'DDP Sync BN Layer.')
    #     self.net = torch.nn.parallel.DistributedDataParallel(
    #         self.net,
    #         device_ids=[self._gpu],
    #         output_device=self._gpu,
    #         broadcast_buffers=False,
    #         #  find_unused_parameters=True
    #     )
    #     logger.success(f'Setting DDP Mode.')
