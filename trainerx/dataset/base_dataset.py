import random
from abc import ABC
from typing import Optional, Callable, List, Union, Dict, Tuple
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        root: str,
        wh: Tuple[int, int],
        loader_type: Optional[str] = 'pil',
        img_type: Optional[str] = 'RGB',
        transform: Optional[Callable] = None,  # to samples
        target_transform: Optional[Callable] = None,  # to target
        preload: Optional[bool] = False
    ) -> None:

        self._root = root
        self._wh = wh
        self._preload = preload
        self._loader_type = loader_type
        self._loader = self.get_image_loader(loader_type)

        self._transform = transform
        self._target_transform = target_transform

        self._SUPPORT_IMG_FORMAT = ['.jpg', '.jpeg', '.png']
        self._SUPPORT_IMG_TYPE = ['RGB', 'GRAY']
        self._PADDING_COLOR = (114, 114, 114)

        self._samples = []
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
    def data_size(self) -> int:
        return len(self._samples)

    def set_transform(self, val) -> None:
        self._transform = val

    def set_target_transform(self, val) -> None:
        self._target_transform = val

    def expanding_data(self, rate: int = 0):
        assert len(self._samples) != 0, f'samples is empty.'

        if rate == 0:
            return

        new_data = []
        for i in range(rate):
            new_data += self._samples
        random.shuffle(new_data)
        self._samples = new_data

    def check_image_suffix(self, filename: str) -> bool:
        return filename.lower().endswith(tuple(self._SUPPORT_IMG_FORMAT))

    def pil_loader(self, path: str) -> Image.Image:
        img = Image.open(path)

        if self.img_type == 'RGB':
            if img.mode != self.img_type:
                img = img.convert(self.img_type)

        if self.img_type == 'GRAY':
            if img.mode != 'L':
                img = img.convert('L')

        return img

    def opencv_loader(self, path: str) -> np.ndarray:
        im = cv2.imread(path)

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

    def label2idx(self, label: str) -> int:
        assert len(self._labels) >= 0, 'labels is empty.'
        assert label in self._labels, 'name not in labels'
        return self._labels.index(label)

    def idx2label(self, idx: int) -> str:
        assert len(self._labels) >= 0, 'labels is empty.'
        assert 0 <= idx <= len(self._labels), '0 <= idx <= len(labels)'
        return self._labels[idx]

    def letterbox(self, image: np.ndarray, pad_color: Optional[tuple] = None) -> tuple:

        pad_color = self._PADDING_COLOR if pad_color is None else pad_color
        src_h, src_w = image.shape[:2]
        dst_w, dst_h = self._wh
        scale = min(dst_h / src_h, dst_w / src_w)
        pad_h, pad_w = int(round(src_h * scale)), int(round(src_w * scale))

        if image.shape[0:2] != (pad_w, pad_h):
            out = cv2.resize(image, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
        else:
            out = image

        top = int((dst_h - pad_h) / 2)
        down = int((dst_h - pad_h + 1) / 2)
        left = int((dst_w - pad_w) / 2)
        right = int((dst_w - pad_w + 1) / 2)

        # add border
        out = cv2.copyMakeBorder(out, top, down, left, right, cv2.BORDER_CONSTANT, value=pad_color)

        x_offset, y_offset = max(left, right) / dst_w, max(top, down) / dst_h
        return out, x_offset, y_offset
