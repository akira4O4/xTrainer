import os
from abc import ABC
from typing import Optional, Callable, List, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from xtrainer.utils.labels import Labels


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        root: str,
        loader_type: Optional[str] = 'opencv',
        transform: Optional[Callable] = None,  # to samples
        target_transform: Optional[Callable] = None,  # to target
    ) -> None:

        assert os.path.exists(root) is True, f'root is not found.'
        self.root = root

        self.img_loader = self.get_image_loader(loader_type)

        self.transform = transform
        self.target_transform = target_transform

        self.samples = []
        self.samples_map: List[int] = []
        self.labels: Labels = None  # noqa

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    @staticmethod
    def opencv_loader(path: str) -> np.ndarray:
        im = cv2.imread(path)
        if im is None:
            raise FileNotFoundError(f'Don`t open image: {path}')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def get_image_loader(self, loader_type: str) -> Callable:
        if loader_type == 'opencv':
            return self.opencv_loader
        elif loader_type == 'pil':
            return self.pil_loader
        else:
            raise ValueError(f'不支持的加载器类型: {loader_type}')
