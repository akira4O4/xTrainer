import os
import random
from typing import Optional, Callable, Tuple, List

import numpy as np
import torch
from PIL import Image
from loguru import logger
from tqdm import tqdm

from trainerx.dataset import Image
from trainerx.dataset.base import BaseDataset
from trainerx.utils.common import get_images
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, labels: torch.Tensor, n_classes: int, n_samples: int):
        super(BalancedBatchSampler, self).__init__(data_source=None)
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0] for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        # This dict keeps track of how many samples have been used for each label
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= self.n_dataset:
            classes = np.random.choice(self.labels_set, size=self.n_classes, replace=False)
            indices = []

            for class_ in classes:
                start_idx = self.used_label_indices_count[class_]
                end_idx = start_idx + self.n_samples
                indices.extend(self.label_to_indices[class_][start_idx:end_idx])
                self.used_label_indices_count[class_] += self.n_samples

                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield indices

            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size


class ClassificationDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        wh: Tuple[int, int],
        loader_type: Optional[str] = 'opencv',
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
            image.data = self._load_image(image.path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample_idx = self._samples_map[index]

        image: Image
        label: int
        image, label = self._samples[sample_idx]

        im = image.data if self._is_preload else self._load_image(image.path)

        if self._transform is not None:
            im = self._transform(im)

        if self._target_transform is not None:
            label = self._target_transform(label)

        return im, label

    def __len__(self) -> int:
        return len(self._samples_map)


if __name__ == '__main__':
    from trainerx.core.preprocess import ClsImageT, ClsTargetT
    from torch.utils.data import DataLoader
    from time import time
    import cv2

    wh = (224, 224)
    t1 = time()
    ds = ClassificationDataset(
        root=r'D:\llf\dataset\danyang\training_data\G\classification\nc3\train\2_youwu',
        wh=wh,
        # is_preload=True,
        transform=ClsImageT(wh),
        target_transform=ClsTargetT()
    )
    dl = DataLoader(ds, 8)
    for imgs, target in dl:
        print(imgs.shape)
        print(target)

    t2 = time()
    print(t2 - t1)
