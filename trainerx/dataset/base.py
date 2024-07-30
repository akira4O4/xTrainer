import os
from abc import ABC
from typing import Optional, Callable, List, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        root: str,
        wh: Tuple[int, int],
        loader_type: Optional[str] = 'opencv',
        img_type: Optional[str] = 'RGB',
        transform: Optional[Callable] = None,  # to samples
        target_transform: Optional[Callable] = None,  # to target
        is_preload: Optional[bool] = False
    ) -> None:

        assert os.path.exists(root) is True, f'root is not found.'

        self._root = root
        self._wh = wh
        self._hw = (wh[1], wh[0])
        self._is_preload = is_preload
        self._loader_type = loader_type
        self._load_image = self.get_image_loader(loader_type)

        self._transform = transform
        self._target_transform = target_transform

        self._SUPPORT_IMG_FORMAT = ['.jpg', '.jpeg', '.png']
        self._SUPPORT_IMG_TYPE = ['RGB', 'GRAY']
        self._PADDING_COLOR = (114, 114, 114)

        self._samples = []
        self._samples_map: List[int] = []
        self._labels: List[str] = []

        assert img_type in self._SUPPORT_IMG_TYPE, 'Image type is not support.'
        self.img_type = img_type

    @property
    def labels(self) -> list:
        return self._labels

    @property
    def num_of_label(self) -> int:
        return len(self._labels)

    @property
    def real_data_size(self) -> int:
        return len(self._samples)

    def set_transform(self, val) -> None:
        self._transform = val

    def set_target_transform(self, val) -> None:
        self._target_transform = val

    def expanding_data(self, rate: int = 0) -> None:
        assert len(self._samples_map) != 0, f'Data samples is empty.'
        assert rate != 0, f'rate == 0.'
        self._samples_map *= rate

    def pil_loader(self, path: str) -> Image.Image:
        img = Image.open(path)

        if self.img_type == 'RGB' and img.mode != 'RGB':
            img = img.convert('RGB')
        elif self.img_type == 'GRAY' and img.mode != 'L':
            img = img.convert('L')

        return img

    def opencv_loader(self, path: str) -> np.ndarray:
        im = cv2.imread(path)
        if im is None:
            raise FileNotFoundError(f'Don`t open image: {path}')

        if self.img_type == 'RGB':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.img_type == 'GRAY':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        return im

    def get_image_loader(self, loader_type: str) -> Callable:
        if loader_type == 'opencv':
            return self.opencv_loader
        elif loader_type == 'pil':
            return self.pil_loader
        else:
            raise ValueError(f'不支持的加载器类型: {loader_type}')

    def label2idx(self, label: str) -> int:
        if not self._labels:
            raise ValueError('标签列表为空。')
        if label not in self._labels:
            raise ValueError(f'标签 {label} 不在标签列表中。')
        return self._labels.index(label)

    def idx2label(self, idx: int) -> str:
        if not self._labels:
            raise ValueError('标签列表为空。')
        if not (0 <= idx < len(self._labels)):
            raise ValueError(f'索引 {idx} 超出标签列表范围。')
        return self._labels[idx]

    def preload(self):
        """此方法需在子类中实现以适应具体的数据加载需求"""
        pass
