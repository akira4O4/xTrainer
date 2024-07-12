import os
from typing import Optional, Callable, List, Union, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from trainerx.dataset import Image, Label
from trainerx.dataset.base import BaseDataset
from trainerx.core.preprocess import letterbox
from trainerx.utils.common import (
    load_json,
    get_images,
    check_size,
    pil_to_np
)


class SegmentationDataSet(BaseDataset):
    def __init__(
        self,
        root: str,
        wh: Tuple[int, int],
        loader_type: Optional[str] = 'pil',
        img_type: Optional[str] = 'RGB',
        transform: Optional[Callable] = None,  # to samples
        target_transform: Optional[Callable] = None,  # to target
        expanding_rate: Optional[int] = 0,
        preload: Optional[bool] = False
    ) -> None:
        super(SegmentationDataSet, self).__init__(
            root=root,
            wh=wh,
            loader_type=loader_type,
            img_type=img_type,
            transform=transform,
            target_transform=target_transform,
            preload=preload
        )

        self._labels = ['0_background_']

        self.samples_with_label: List[Tuple[Image, Label]] = []
        self.background_samples: List[Tuple[Image, Label]] = []

        self.all_image_path: List[str] = get_images(self._root, self._SUPPORT_IMG_FORMAT)

        self.load_data()

        if self._preload:
            self.preload_mask()

        self._samples = self.samples_with_label + self.background_samples

        if expanding_rate != 0:
            self.expanding_data(expanding_rate)

    @staticmethod
    def find_label_path(path: str) -> str:
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        label_path = path.replace(ext, '.json')
        return label_path

    # Load data path
    def load_data(self) -> None:
        logger.info('loading dataset...')
        for image_path in tqdm(self.all_image_path):

            label = Label()
            image = Image(path=image_path)

            label_path: str = self.find_label_path(image_path)
            if os.path.exists(label_path):

                # load and decode json data
                label.load_metadata(load_json(label_path))

                # check image path
                if os.path.basename(image_path) != label.image_path:
                    logger.warning(f'img_path != json.imagePath,{image_path}')
                    continue

                self.samples_with_label.append((image, label))

                # find the all label
                for obj in label.objects:
                    if obj["label"] not in self._labels:
                        self._labels.append(obj["label"])
            else:
                self.background_samples.append((image, label))

        self._labels.sort()

        logger.info(f'samples_with_label: {len(self.samples_with_label)}')
        logger.info(f'background_samples: {len(self.background_samples)}')

    # Preload mask but don`t preprocessing mask
    def preload_mask(self) -> None:
        logger.info('Preload mask...')

        image: Image
        label: Label
        for image, label in tqdm(self.samples_with_label):
            image.data = self._loader(image.path)
            label.mask = self.get_mask(label.objects)

        for image, label in tqdm(self.background_samples):
            image.data = self._loader(image.path)
            label.mask = np.zeros(self._hw, dtype=np.uint8)

    def get_mask(self, objects: list) -> np.ndarray:

        mask = np.zeros(self._hw, dtype=np.uint8)

        obj: dict  # obj:{"label":"","points":[]}
        for obj in objects:
            # points:[pt1,pt2,...]
            points: list = obj["points"]
            label: str = obj['label']
            mask = self.polygon2mask(mask, points, self.label2idx(label))
        return mask

    @staticmethod
    def polygon2mask(mask: np.ndarray, points: list, label_idx: Optional[int] = 0) -> np.ndarray:
        # points=[[x,y],[x,y],...]
        assert 255 >= label_idx >= 0, '255 >= label_idx >= 0'
        points = np.asarray(points, dtype=np.int32)
        cv2.fillConvexPoly(mask, points, color=label_idx)  # noqa
        return mask

    # TODO: Test code
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        image, label = self._samples[index]

        im = self._loader(image.path) if self._preload else image.data
        mask = self.get_mask(label.objects) if self._preload else label.mask

        if not check_size(im, self._wh):
            im, x_offset, y_offset = letterbox(im, self._wh)

        if self._transform is not None:
            im = self._transform(im)

        if self._target_transform is not None and not label.is_background:
            mask = self._target_transform(mask)

        mask = mask[None]  # (h, w) -> (1, h, w)
        return im, mask  # noqa

    def __len__(self) -> int:
        return len(self._samples)


if __name__ == '__main__':
    ds = SegmentationDataSet(root=r'D:\llf\dataset\danyang\training_data\E\seg', wh=(960, 960), preload=True)
    print(ds.num_of_label)
    print(ds.labels)
