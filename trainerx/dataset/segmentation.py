import os
from typing import Optional, Callable, List, Union, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from trainerx.dataset import Image, SegLabel
from trainerx.dataset.base import BaseDataset
from trainerx.core.preprocess import LetterBox
from trainerx.utils.torch_utils import npimage2torch, np2torch
from trainerx.utils.common import (
    load_json,
    get_images,
    get_image_shape,
    np2pil,
    pil2np
)


class SegmentationDataSet(BaseDataset):
    def __init__(
        self,
        root: str,
        wh: Tuple[int, int],
        loader_type: Optional[str] = 'opencv',
        img_type: Optional[str] = 'RGB',
        transform: Optional[Callable] = None,  # to samples
        # target_transform: Optional[Callable] = None,  # to target
        expanding_rate: Optional[int] = 1,
        is_preload: Optional[bool] = False
    ) -> None:
        super(SegmentationDataSet, self).__init__(
            root=root,
            wh=wh,
            loader_type=loader_type,
            img_type=img_type,
            transform=transform,
            # target_transform=target_transform,
            is_preload=is_preload
        )

        self._labels = ['0_background_']

        self.samples_with_label: List[Tuple[Image, SegLabel]] = []
        self.background_samples: List[Tuple[Image, SegLabel]] = []

        self.all_image_path: List[str] = get_images(self._root, self._SUPPORT_IMG_FORMAT)

        self.load_data()

        if self._is_preload:
            self.preload()

        self._samples = self.samples_with_label + self.background_samples
        self._samples_map: List[int] = list(range(len(self._samples)))

        self.expanding_data(expanding_rate)
        self.letterbox = LetterBox(wh)

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

            label = SegLabel()
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
    def preload(self) -> None:
        logger.info('Preload mask...')

        image: Image
        label: SegLabel
        for image, label in tqdm(self.samples_with_label):
            image.data = self._loader(image.path)
            label.mask = self.get_mask(label.objects)

        for image, label in tqdm(self.background_samples):
            image.data = self._loader(image.path)
            label.mask = np.zeros(self._hw, dtype=np.uint8)

    def get_mask(self, objects: list) -> np.ndarray:

        mask = np.zeros(self._hw, dtype=np.uint8)

        obj: dict  # obj:{"label":"","points":[]}

        if objects is None or objects == []:
            return mask

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
        sample_idx = self._samples_map[index]

        image: Image
        label: SegLabel
        image, label = self._samples[sample_idx]

        im = image.data if self._is_preload else self._loader(image.path)
        mask = label.mask if self._is_preload else self.get_mask(label.objects)

        img_h, img_w = get_image_shape(im)
        assert img_w > 0 and img_h > 0, f'Error: img_w or img_h <=0'

        # input.type=[Image.Image,torch.Tensor]
        im, mask = self._transform((im, mask))

        mask = mask[None]  # (h, w) -> (1, h, w)
        return im, mask

    def __len__(self) -> int:
        return len(self._samples)
