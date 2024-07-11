import os
import random
from typing import Optional, Callable, Tuple, List, Union, Dict

import torch
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm
from .base_dataset import BaseDataset
from ..utils.common import get_images


class ClassificationDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        wh: Tuple[int, int],
        loader_type: Optional[str] = 'pil',
        img_type: Optional[str] = 'RGB',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        expanding_rate: Optional[int] = 0,
        preload: Optional[bool] = False
    ):
        super(ClassificationDataset, self).__init__(
            root=root,
            wh=wh,
            loader_type=loader_type,
            img_type=img_type,
            transform=transform,
            target_transform=target_transform,
            preload=preload
        )
        self._memory: Dict[str, np.ndarray] = {}
        self.find_labels()

        # samples=[('xxx/xxx.jpg',1),(xxx/xxx.jpg,0),...]
        self._samples: List[Tuple[str, int]] = []
        self.load_data()

        if expanding_rate > 0:
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

        # samples=[('xxx/xxx.jpg',1),(xxx/xxx.jpg,0),...]
        for idx in range(self.num_of_label):
            target_path = os.path.join(self._root, self._labels[idx])

            images: List[str] = get_images(target_path, self._SUPPORT_IMG_FORMAT)

            self._samples.extend(list(map(lambda x: (x, idx), images)))  # noqa

        random.shuffle(self._samples)

        # if self._load_all_data:
        #     self.load_all_data_to_memory()

    def load_all_data_to_memory(self) -> None:
        logger.info(f'load all data to memory...')
        for path in tqdm(self._samples):
            image: Union[Image.Image, np.ndarray] = self._loader(path)
            self._memory[path] = image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:

        path, label_idx = self._samples[index]

        if self._preload:
            image = self._memory.get(path)
        else:
            image: Union[Image.Image, np.ndarray] = self._loader(path)

        img_w: int = -1
        img_h: int = -1

        if isinstance(image, Image.Image):
            img_w, img_h = image.size
        elif isinstance(image, np.ndarray):
            img_h, img_w = image.shape

        assert img_w > 0 and img_h > 0, f'Error: img_w or img_h <=0'

        if img_h != self._wh[1] or img_w != self._wh[0]:
            if isinstance(image, Image.Image):
                # PIL.Image -> numpy.ndarray
                image = np.asarray(image)  # noqa

            image, _, _ = self.letterbox(image)

        image: torch.Tensor
        if self._transform is not None:
            image = self._transform(image)

        if self._target_transform is not None:
            label_idx = self._target_transform(label_idx)

        return image, label_idx

    def __len__(self) -> int:
        return len(self._samples)
