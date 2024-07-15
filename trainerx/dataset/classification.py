import os
import random
from typing import Optional, Callable, Tuple, List

import numpy as np
import torch
from PIL import Image
from loguru import logger
from tqdm import tqdm

from trainerx.utils.common import get_images
from trainerx.dataset.base import BaseDataset
from trainerx.dataset import Image
from trainerx.core.preprocess import letterbox


class ClassificationDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        wh: Tuple[int, int],
        loader_type: Optional[str] = 'pil',
        img_type: Optional[str] = 'RGB',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        expanding_rate: Optional[int] = 1,
        is_preload: Optional[bool] = False
    ):
        super(ClassificationDataset, self).__init__(
            root=root,
            wh=wh,
            loader_type=loader_type,
            img_type=img_type,
            transform=transform,
            target_transform=target_transform,
            is_preload=is_preload
        )
        self.find_labels()

        # samples=[(Image,1),(Image,0),...]
        self._samples: List[Tuple[Image, int]] = []

        logger.info(f'Load data ...')
        self.load_data()
        if self._is_preload:
            logger.info(f'Preload image data ...')
            self.preload()

        self._samples_map: List[int] = list(range(len(self._samples)))

        self.expanding_data(expanding_rate)

        self.targets = [s[1] for s in self._samples]
        if len(self._samples) == 0:
            logger.warning(f"Found 0 files in sub folders of: {self._root}\n")

    def find_labels(self) -> None:
        for d in os.scandir(self._root):
            if d.is_dir():
                self._labels.append(d.name)
        self._labels.sort()

    def load_data(self) -> None:
        for idx in tqdm(range(self.num_of_label)):
            target_path = os.path.join(self._root, self.idx2label(idx))

            images: List[str] = get_images(target_path, self._SUPPORT_IMG_FORMAT)

            self._samples.extend(list(map(lambda x: (Image(path=x), idx), images)))  # noqa

        random.shuffle(self._samples)

    def preload(self) -> None:
        image: Image
        for image, idx in tqdm(self._samples):
            image.data = self._loader(image.path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample_idx = self._samples_map[index]

        image: Image
        label: int
        image, label = self._samples[sample_idx]

        im = image.data if self._is_preload else self._loader(image.path)

        img_w: int = -1
        img_h: int = -1

        if isinstance(im, Image.Image):
            img_w, img_h = im.size
        elif isinstance(im, np.ndarray):
            img_h, img_w = im.shape

        assert img_w > 0 and img_h > 0, f'Error: img_w or img_h <=0'

        if img_h != self._wh[1] or img_w != self._wh[0]:
            if isinstance(im, Image.Image):
                # PIL.Image -> numpy.ndarray
                im = np.asarray(im)  # noqa

            im, _, _ = letterbox(im, self._wh)

        im: torch.Tensor
        if self._transform is not None:
            im = self._transform(im)

        if self._target_transform is not None:
            label = self._target_transform(label)

        return im, label

    def __len__(self) -> int:
        return len(self._samples_map)
