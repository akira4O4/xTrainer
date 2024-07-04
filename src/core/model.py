import os
import os.path as osp
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
        use_ddp: bool = False,
    ):
        self._model_name = model_name
        self._pretrained = pretrained
        self._num_classes = num_classes
        self._mask_classes = mask_classes
        self._weight = weight
        self._strict = strict
        self._device = torch.device('cpu')
        self._map_location = map_location
        self._use_ddp = use_ddp

        if device != -1 and torch.cuda.is_available():
            self._device = torch.device(f'cuda:{device}')
        else:
            self._device = torch.device('cpu')

        self.net = None

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    @property
    def training(self) -> bool:
        return self.net.training

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
        return self.net.state_dict()

    @property
    def device(self) -> torch.device:
        return self._device

    def set_gpu(self, gpu: int) -> None:
        self._gpu = gpu
        self._device = f'cuda:{gpu}'
        logger.info(f'Change gpu to {self._gpu}')
        logger.info(f'Change device to {self._device}')

    @property
    def parameters(self):
        return self.net.parameters()

    def train(self) -> None:
        self.net.train()

    def eval(self) -> None:
        self.net.eval()

    def move_to_device(self) -> None:
        self.net.to(self._device)
        logger.info(f'Move model.to: {self._device}.')

    def set_model_path(self, path: str) -> None:
        self._weight = path

    def init(self) -> None:

        self.net = self.create_model(
            self._model_name,
            num_classes=self._num_classes,
            mask_classes=self._mask_classes,
            pretrained=self._pretrained,
        )

        if osp.exists(self._weight) is True:
            self.load_weight(self._weight, self._strict, self._map_location)
            logger.success('Load model done.')
        else:
            logger.warning(f'Model path:{self._weight} is not found.')

        logger.info(f'Current device is: {self._device}.')
        logger.success(f'Build model success.')

    @staticmethod
    def create_model(
        model_name: str,
        num_classes: Optional[int] = 0,
        mask_classes: Optional[int] = 0,
        pretrained: Optional[bool] = False,
        input_channels: Optional[int] = 3,  # default RGB
        **kwargs
    ):
        if num_classes == 0 and mask_classes == 0:
            logger.error(f'Cannot [equal 0] num_classes and mask_classes at the same time')
            raise

        net = network.__dict__.get(model_name)
        if net is None:
            logger.error(f'Don`t get the model:{model_name}.')
            exit()

        return net(
            num_classes=num_classes,
            mask_classes=mask_classes,
            pretrained=pretrained,
            input_channels=input_channels,
            **kwargs
        )

    def load_weight(
        self,
        weight_path: str,
        strict: Optional[bool] = True,
        map_location: Optional[str] = 'cpu',
    ) -> None:

        if not osp.exists(weight_path):
            logger.warning(f"{weight_path} 权重文件不存在。")
            return

        weight = torch.load(weight_path, map_location=map_location)
        model_state_dict = self.net.state_dict()
        weight_state_dict = weight.get('state_dict')

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

        self.net.load_state_dict(model_state_dict, strict=strict)

        logger.info(f'Loading [{loading_item}/{total_item}] item to model.')
        logger.success(f'Loading :{weight_path} .')

    def save_checkpoint(
        self,
        save_path: str,
        epoch: Optional[int] = 0,
        other_kw: Optional[dict] = None,
    ) -> None:

        if not osp.exists(save_path):
            os.makedirs(save_path)
            logger.success(f"Create path {save_path} to save weight.")

        # Remove DDP key

        if self._use_ddp:
            state_dict = self.remove_ddp_key()
        else:
            state_dict = self.state_dict

        save_dict = {
            "state_dict": state_dict,
        }

        if other_kw is not None:
            for k, v in other_kw.items():
                save_dict.update({k: v})

        curr_time = get_time()
        model_save_path = os.path.join(save_path, f"{curr_time}_epoch{epoch}.pth")  # e.g.20240101_epoch1.pth
        torch.save(save_dict, model_save_path)
        logger.success(f'👍 Save epoch:{epoch} weight to :{save_path}\n')

    def remove_ddp_key(self) -> dict:
        warnings.warn("", DeprecationWarning)
        del_count = 0
        del_ddp_key_flag = False
        new_state_dict = OrderedDict()
        for k, v in self.state_dict.items():
            if 'module' in k:
                # k = k[7:]
                k = k.replace('module.', '')
                del_count += 1
                del_ddp_key_flag = True

            new_state_dict[k] = v

        if del_ddp_key_flag:
            logger.info(f'Delete num of key(module) of DDP model: {del_count} item.')

        return new_state_dict

    def ddp_mode(self, sync_bn: bool = False) -> None:

        warnings.warn("", DeprecationWarning)
        if self._gpu is None or self._gpu == -1:
            logger.warning(f'Your GPU:{self._gpu}.')
            return

        if self.net is None:
            logger.error(f'Model is not init.')
            return

        if sync_bn:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            logger.info(f'DDP Sync BN Layer.')
        self.net = torch.nn.parallel.DistributedDataParallel(
            self.net,
            device_ids=[self._gpu],
            output_device=self._gpu,
            broadcast_buffers=False,
            #  find_unused_parameters=True
        )
        logger.success(f'Setting DDP Mode.')
