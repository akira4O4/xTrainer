import os
from typing import Optional, Callable, List, Union, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger
from tqdm import tqdm

from trainerx.dataset.base import BaseDataset
from trainerx.dataset import Img, Label
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

        self.load_data()

        self.samples_with_label: List[Tuple[Img, Label]] = []
        self.background_samples: List[Tuple[Img, Label]] = []
        self._samples = self.samples_with_label + self.background_samples

        if expanding_rate != 0:
            self.expanding_data(expanding_rate)

    @staticmethod
    # find label path from img
    def find_label_path(path: str) -> str:
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        label_path = path.replace(ext, '.json')
        return label_path

    def load_data(self) -> None:

        logger.info('loading dataset...')
        images_path: List[str] = get_images(self._root, self._SUPPORT_IMG_FORMAT)

        for image_path in tqdm(images_path):

            # preload image
            if self._preload:
                image = Img(path=image_path, image=self._loader(image_path))
            else:
                image = Img(path=image_path)

            label = Label()
            label_path: str = self.find_label_path(image_path)
            if os.path.exists(label_path):

                label.load_metadata(load_json(label_path))

                # check image path
                if image_path != label.image_path:
                    logger.warning(f'img_path != json.imagePath,{image_path}')
                    continue

                # preload mask
                if self._preload:
                    label.mask = self.get_mask(label.objects)

                self.samples_with_label.append((image, label))

                # find the all label
                for obj in label.objects:
                    if obj["label"] not in self._labels:
                        self._labels.append(obj["label"])

            else:
                # preload background mask
                if self._preload:
                    label.mask = np.zeros(self._hw, dtype=np.uint8)

                self.background_samples.append((image, label))

        self._labels.sort()

        logger.info(f'samples_with_label: {len(self.samples_with_label)}')
        logger.info(f'background_samples: {len(self.background_samples)}')

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

        img, label = self._samples[index]
        if self._preload:
            image = img.image
        else:
            image = self._loader(img.path)

        mask: np.ndarray = self.get_mask(label.objects) if self._preload else label.mask

        if not check_size(image, self._wh):
            image, x_offset, y_offset = self.letterbox(image)

        if self._transform is not None:
            image = self._transform(image)

        if self._target_transform is not None and not label.is_background:
            mask = self._target_transform(mask)

        mask = mask[None]  # (h, w) -> (1, h, w)
        return image, mask  # noqa

    def __len__(self) -> int:
        return len(self._samples)


if __name__ == '__main__':
    ...
