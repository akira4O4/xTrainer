import os
import os.path as osp
import shutil
from typing import Optional
from collections import OrderedDict

import torch
from loguru import logger

import network
from utils.util import get_time, split_line

__all__ = ['Model']


class Model:
    def __init__(
            self,
            model_name: str,
            num_classes: Optional[int] = 0,
            mask_classes: Optional[int] = 0,
            pretrained: Optional[bool] = False,
            model_path: Optional[str] = None,
            gpu: Optional[int] = 0,  # -1==cpu
            strict: Optional[bool] = True,
            map_location: Optional[str] = 'cpu',
            input_channels: Optional[str] = 3,
            use_ddp: bool = False,
    ):
        self._model_name = model_name
        self._pretrained = pretrained
        self._num_classes = num_classes
        self._mask_classes = mask_classes
        self._model_path = model_path
        self._strict = strict
        self._gpu = gpu
        self._map_location = map_location
        self._input_channels = input_channels
        self._use_ddp = use_ddp

        if gpu != -1 and torch.cuda.is_available():
            self._device = torch.device(f'cuda:{gpu}')
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

    @property
    def gpu(self) -> int:
        return self._gpu

    @gpu.setter
    def gpu(self, gpu: int) -> None:
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

    # @model_path.setter
    # def model_path(self, path: str):
    #     self._model_path = path

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

    def load_model(
            self,
            model_path: str,
            strict: Optional[bool] = True,
            map_location: Optional[str] = 'cpu',
    ) -> None:

        if not osp.exists(model_path):
            logger.warning(f"{model_path} 权重文件不存在。")
            return

        checkpoint = torch.load(model_path, map_location=map_location)
        model_state_dict = self.net.state_dict()
        pretrain_model_state_dict = checkpoint.get('state_dict')

        if pretrain_model_state_dict is None:
            logger.error(f'checkpoint do not found the model state_dict.')
            return

        total_item = 0
        loading_itme = 0

        for k, v in pretrain_model_state_dict.items():
            total_item += 1

            if 'module' in k:  # DDP model
                k = k[7:]

            if k in model_state_dict.keys():
                if v.shape == model_state_dict[k].shape:
                    model_state_dict[k] = v
                    loading_itme += 1

        self.net.load_state_dict(model_state_dict, strict=strict)

        logger.info(f'Loading :{model_path} .')
        logger.info(f'Loading [{loading_itme}/{total_item}] item to model.')

    def init_model(self) -> None:

        self.net = self.create_model(
            self._model_name,
            num_classes=self._num_classes,
            mask_classes=self._mask_classes,
            pretrained=self._pretrained,
        )

        if osp.exists(self._model_path) is True:
            self.load_model(self._model_path, self._strict, self._map_location)
            logger.success('Load model done.')
        else:
            logger.warning(f'Model path:{self._model_path} is not found.')

        logger.info(f'Build model done.')
        logger.info(f'Current model.net device is: {self._device}.')

    def save_checkpoint(
            self,
            save_path: str,
            epoch: Optional[int] = 0,
            model_info: Optional[dict] = None
    ) -> None:

        if not osp.exists(save_path):
            os.makedirs(save_path)
            logger.success(f"Create path {save_path} to save checkpoint.")

        # Remove DDP key

        if self._use_ddp:
            state_dict = self.remove_ddp_key()
        else:
            state_dict = self.state_dict

        save_dict = {
            "epoch": epoch,
            "state_dict": state_dict,
        }

        model_info_str = ''
        if model_info is not None:
            for k, v in model_info.items():
                model_info_str += f'_{k}{v}'

        curr_time = get_time()
        model_save_path = os.path.join(save_path, f"{curr_time}_Epoch{epoch}{model_info_str}.pth")
        torch.save(save_dict, model_save_path)
        logger.success(f'👍 Epoch:{epoch}: Save the Weight to :{save_path}')

    def remove_ddp_key(self) -> dict:
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
