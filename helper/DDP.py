import os
from typing import Optional, List

import numpy as np
import torch
# from torch.backends import cudnn
import torch.utils.data
from PIL import ImageFile
from loguru import logger
# from bunch import Bunch
import torch.distributed as dist

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['DDPWrapper']

Tensor = torch.Tensor
ListTensor = List[torch.Tensor]


class DDPWrapper:
    def __init__(
        self,
        backend: Optional[str] = 'nccl',
        init_method: Optional[str] = 'env://',
        **kwargs
    ):
        self._init_method = init_method
        self._backend = backend

        self._rank = None
        self._local_rank = None
        self._world_size = None
        self._cuda_visible_devices = None
        self._master_addr = None
        self._master_port = None
        self._gpu = None
        self._device = None
        self._backend = kwargs.get('backend')
        self._init_method = kwargs.get('init_method')
        self._batch_size = kwargs.get('batch_size')
        self._sync_bn = kwargs.get('sync_bn')
        self._num_workers = kwargs.get('num_workers')
        # self._sync_performance = kwargs.get('sync_performance')
        self.kwargs = kwargs

    def init_distributed_mode(self):
        self.get_env_params()
        self.show_env_params()
        logger.info(f'Backend:{self._backend}')
        logger.info(f'Init_Method:{self._init_method}')

        dist.init_process_group(
            backend=self._backend,
            init_method=self._init_method,
            world_size=self._world_size,
            rank=self._rank
        )

        # torch.cuda.set_device(self._gpu)
        logger.success(f'{self._device} is init Done.')
    
    @property
    def num_workers(self) -> int:
        return self._num_workers

    # @property
    # def sync_performance(self) -> bool:
    #     return self._sync_performance

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def init_method(self) -> str:
        return self._init_method

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def sync_bn(self) -> bool:
        return self._sync_bn

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def master_addr(self):
        return self._master_addr

    @property
    def master_port(self):
        return self._master_port

    @property
    def cuda_visible_devices(self):
        return self._cuda_visible_devices

    @property
    def gpu(self) -> int:
        return self._gpu

    @property
    def device(self) -> torch.device:
        return self._device

    def get_env_params(self) -> None:
        self._rank = int(os.environ.get('RANK'))
        self._local_rank = int(os.environ['LOCAL_RANK'])
        self._world_size = int(os.environ['WORLD_SIZE'])
        self._master_addr = os.environ['MASTER_ADDR']
        self._master_port = int(os.environ['MASTER_PORT'])
        self._cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(
            ',')
        self._gpu = self._local_rank
        self._device = torch.device(f"cuda:{self._gpu}")

    @staticmethod
    def show_env_params():
        logger.info(
            f"|| MASTER_ADDR:{os.environ.get('MASTER_ADDR')}"
            f"|| MASTER_PORT{os.environ.get('MASTER_PORT')}"
            f"|| LOCAL_RANK:{os.environ.get('LOCAL_RANK')}"
            f"|| RANK:{os.environ.get('RANK')}"
            f"|| WORLD_SIZE:{os.environ.get('WORLD_SIZE')}"
            f"|| CUDA_VISIBLE_DEVICES:{os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

    def get_distributed_sampler(self, dataset):
        self._distributed_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
        return self._distributed_sampler

    def sync_tensor(self, data, device: torch.device) -> list:
        # Return List[Constant]
        input_data = []

        if isinstance(data, list):
            input_data = data
        elif isinstance(data, tuple):
            input_data = list(data)
        else:
            input_data.append(data)

        sync_data = []

        for data in input_data:
            result = self._sync_tensor_impl(data, device)
            # sync_data.append(result[0])
            sync_data.append(result)

        return sync_data

    @staticmethod
    def _sync_tensor_impl(data, device: torch.device):
        # Input: data.type=  Constant or Tensor
        # Return:data.type=  Constant
        if isinstance(data, Tensor):
            result = data.to(device)
        else:
            result = torch.tensor([data], device=device)

        # logger.debug(result)
        dist.barrier()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        # logger.debug(result.tolist())
        # result = result.tolist()
        result = result.item()
        return result
