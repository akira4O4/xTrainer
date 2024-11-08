import os
import random
from collections import defaultdict
from typing import Optional, Callable, Tuple, List

import numpy as np
import torch
import torchvision
from loguru import logger
from tqdm import tqdm
from torch.utils.data import Sampler

from xtrainer.dataset import Image
from xtrainer.utils.labels import Labels
from xtrainer.dataset.base import BaseDataset
from xtrainer.utils.common import get_images
from xtrainer.augment.functional import letterbox


class ClassificationDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        loader_type: Optional[str] = 'opencv',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        repeat: Optional[int] = 1,
        cache: Optional[bool] = False
    ):
        super(ClassificationDataset, self).__init__(
            root=root,
            loader_type=loader_type,
            transform=transform,
            target_transform=target_transform,
        )

        # Get the data
        self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples

        if len(self.samples) == 0:
            logger.error(f"Found 0 files in sub folders of: {self.root}\n")
            return

        self.repeated_data(repeat)
        self.targets = [s[1] for s in self.samples]
        random.shuffle(self.samples)

    def repeated_data(self, rate: int) -> None:
        if rate > 1:
            self.samples *= rate

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        im, label = self.samples[index]

        if self.transform is not None:
            im = self.transform(im)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return im, label

    def __len__(self) -> int:
        return len(self.samples)


if __name__ == '__main__':
    dataset = ClassificationDataset(
        root=r'D:\llf\dataset\mnist\images\train',
        loader_type='opencv',
        repeat=2
    )
    print(f'data size: {len(dataset)}')
    im, label = dataset[0]
    print(label)
