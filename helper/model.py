import os
import os.path as osp
import shutil
from typing import Optional
from collections import OrderedDict

import torch
from loguru import logger

import network
from utils.util import get_time

__all__ = ['Model']


class Model:
    def __init__(
            self,
            model_name: str,
            num_classes: Optional[int] = 0,
            mask_classes: Optional[int] = 0,
            pretrained: Optional[bool] = False,
            model_path: Optional[str] = None,
            gpu: Optional[int] = -1,  # default (-1)==cpu
            strict: Optional[bool] = True,
            map_location: Optional[str] = 'cpu',
            input_channels: Optional[str] = 1,
            **kwargs
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

        if gpu != -1 and torch.cuda.is_available():
            self._device = torch.device(f'cuda:{gpu}')
        else:
            self._device = torch.device('cpu')

        self.net = None

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    def set_gpu(self, gpu: int) -> None:
        self._gpu = gpu
        self._device = f'cuda:{gpu}'
        logger.info(f'Change gpu to {self._gpu}')
        logger.info(f'Change device to {self._device}')

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

    def setting_weight(self, path: str):
        self._model_path = path

    def init_model(self) -> None:
        """
        1.create model
        2.load or not weight
        3.return model
        """
        self.net = self.create_model(
            self._model_name,
            num_classes=self._num_classes,
            mask_classes=self._mask_classes,
            pretrained=self._pretrained,

        )

        if osp.exists(self._model_path) is True:
            self.load_model(self._model_path, self._strict, self._map_location)
        else:
            logger.warning(f'model path:{self._model_path} is not found.')

        logger.info(f'Build model done.')
        logger.info(f'Current model.net device is: {self._device}.')

    @staticmethod
    def create_model(
            model_name: str,
            num_classes: Optional[int] = 0,
            mask_classes: Optional[int] = 0,
            pretrained: Optional[bool] = False,
            input_channels: Optional[int] = 3,
            **kwargs
    ):
        if num_classes == 0 and mask_classes == 0:
            logger.error(
                f'Cannot [equal 0] num_classes and mask_classes at the same time')
            raise

        net = network.__dict__.get(model_name)
        if net is None:
            logger.error(f'Don`t get the model:{model_name}.')
            exit()

        return net(num_classes=num_classes,
                   mask_classes=mask_classes,
                   pretrained=pretrained,
                   input_channels=input_channels,
                   **kwargs)

    def load_model(
            self,
            model_path: str,
            strict: Optional[bool] = True,
            map_location: Optional[str] = 'cpu',
            **kwargs
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
                else:
                    print(total_item, k, v.shape, '->', model_state_dict[k].shape)

        self.net.load_state_dict(model_state_dict, strict=strict)

        logger.info(f'Loading :{model_path} .')
        logger.info(f'Loading [{loading_itme}/{total_item}] item to model.')

    def save_checkpoint(
            self,
            save_path: str,
            epoch: Optional[int] = 0,
            is_best: Optional[bool] = False,
            output_name: Optional[str] = "checkpoint.pth",
            **kwargs
    ) -> None:

        filename = osp.join(save_path, output_name)

        if not osp.exists(save_path):
            os.makedirs(save_path)
            logger.success(f"Create path {save_path} to save checkpoint.")

        # Remove DDP key
        # DDP:  model.module.state_dict()
        # norm: model.state_dict()
        del_ddp_key_flag = False
        del_count = 0
        new_state_dict = OrderedDict()
        for k, v in self.state_dict.items():
            if 'module' in k:
                # k = k[7:]
                k = k.replace('module.', '')
                del_count += 1
                del_ddp_key_flag = True

            new_state_dict[k] = v

        if del_ddp_key_flag:
            logger.info(
                f'Delete num of key(module) of DDP model: {del_count} item.')

        save_dict = {
            "epoch": epoch,
            "state_dict": new_state_dict,
            # 'optimize_state_dict': optimize_state_dict
        }

        torch.save(save_dict, filename)
        logger.success(f'📂 Epoch:{epoch}: Save the checkpoint to :{filename}')

        if is_best is True:
            curr_time = get_time()
            best_model_name = ''

            for item in kwargs.items():
                best_model_name += f'{item[0]}{item[1]}_'

            # e.g.2022-11-14-(17:35:42)_kv_kv_model_best.pth.tar
            shutil.copyfile(filename,
                            os.path.join(save_path, f"{curr_time}_Epoch{epoch}_{best_model_name}BestModel.pth"))
            logger.success(
                f'👍 Epoch:{epoch}: Save the [Best Model] to :{filename.replace("checkpoint", "xxx_BestModel")}')

    def ddp_mode(self, sync_bn: bool = False) -> None:
        if self._gpu is None:
            logger.warning(f'Your GPU:{self._gpu}.')
            return

        if self.net is None:
            logger.error(f'Model is not init.')
            return

        if sync_bn:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            logger.info(f'DDP Sync BN Layer.')
        self.net = torch.nn.parallel.DistributedDataParallel(self.net,
                                                             device_ids=[
                                                                 self._gpu],
                                                             output_device=self._gpu,
                                                             broadcast_buffers=False,
                                                             #  find_unused_parameters=True
                                                             )
        logger.success(f'Setting DDP Mode.')
