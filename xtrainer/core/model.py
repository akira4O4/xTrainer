import os
import platform
from datetime import datetime
from typing import Optional, Any

import torch
from loguru import logger


class Model:
    def __init__(
        self,
        ckpt_path: Optional[str] = None,  # checkpoint path
        device: Optional[int] = 0,  # -1==cpu
        strict: Optional[bool] = False,
        map_location: Optional[str] = 'cpu',
    ):
        self._net = None
        self.checkpoint = {}
        self._strict = strict
        self._map_location = map_location
        self._device = torch.device('cpu')

        # Init
        self.set_device(device)
        self._load_checkpoint(ckpt_path)

    @property
    def training(self) -> bool:
        return self._net.training

    @property
    def state_dict(self) -> dict:
        return self._net.state_dict()

    @property
    def device(self) -> torch.device:
        return self._device

    def set_device(self, device: int) -> None:
        if not torch.cuda.is_available() or platform.system() == 'Windows':
            self._device = torch.device('cpu')
        else:
            if isinstance(device, str):
                self._device = torch.device(device)
            elif isinstance(device, int):
                if device < 0:
                    self._device = torch.device('cpu')
                else:
                    self._device = torch.device(f'cuda:{device}')

    def to_device(self) -> None:
        self._net.to(self._device)

    def set_checkpoint(self, path: str) -> None:
        self._load_checkpoint(path)

    def train(self) -> None:
        self._net.train()

    def eval(self) -> None:
        self._net.eval()

    def set_net(self, val) -> None:
        self._net = val

    def parameters(self):
        return self._net.parameters()

    def __call__(self, images: torch.Tensor) -> Any:

        if self.training:
            return self._net(images)
        else:
            with torch.no_grad():
                return self._net(images)

    def _load_checkpoint(self, path: str) -> None:
        if (path is None) or (os.path.exists(path) is False):
            logger.warning('Weight is not found.')
        else:
            self.checkpoint = torch.load(path, map_location=self._map_location)

    def load_weight(self) -> None:
        if self.checkpoint == {}:
            return

        ckpt_state_dict = self.checkpoint.get('state_dict', None)
        if ckpt_state_dict is None:
            logger.error('Weight.state_dict is not found.')
            return

        if self._net is None:
            logger.warning('Net is None.')
            return

        model_state_dict = self._net.state_dict()

        total_item = 0
        loading_item = 0

        for (net_key, net_val), (weight_key, weight_val) in zip(model_state_dict.items(),
                                                                ckpt_state_dict.items()):
            total_item += 1

            if net_val.shape == weight_val.shape:
                model_state_dict[net_key] = weight_val
                loading_item += 1

        self._net.load_state_dict(model_state_dict, strict=self._strict)

        logger.info(f'Loading [{loading_item}/{total_item}] item to model.')

    def save_checkpoint(self, path: str) -> None:
        # "epoch"
        # "state_dict"
        # "optimizer"
        # "date"
        # "lr"
        self.checkpoint.update({
            "state_dict": self.state_dict,
            "date": datetime.now().isoformat()
        })
        torch.save(self.checkpoint, path)
