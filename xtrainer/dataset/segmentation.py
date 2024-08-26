import os
from typing import Optional, Callable, List, Tuple
import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from xtrainer.dataset import Image
from xtrainer.dataset.base import BaseDataset
from xtrainer.augment.functional import letterbox
from xtrainer.utils.labels import MaskLabel, Labels
from xtrainer.utils.common import (
    load_json,
    get_images,
    get_image_wh,
    hw_to_hw1,
    safe_round
)


class SegmentationDataSet(BaseDataset):
    def __init__(
        self,
        root: str,
        wh: Tuple[int, int],
        labels: Labels,
        loader_type: Optional[str] = 'opencv',
        img_type: Optional[str] = 'RGB',
        transform: Optional[Callable] = None,  # to samples
        expanding_rate: Optional[int] = 1,
        cache: Optional[bool] = False
    ) -> None:
        super(SegmentationDataSet, self).__init__(
            root=root,
            wh=wh,
            loader_type=loader_type,
            img_type=img_type,
            transform=transform,
            cache=cache
        )

        self._labels = labels

        self.samples_with_label: List[Tuple[Image, MaskLabel]] = []
        self.background_samples: List[Tuple[Image, MaskLabel]] = []

        self.all_image_path: List[str] = get_images(self._root, self._SUPPORT_IMG_FORMAT)

        self.load_data()

        if self._use_cache:
            self.cache_images_to_memory()

        self._samples = self.samples_with_label + self.background_samples  # [(image,label),(image,label),...]
        self._samples_idx_map: List[int] = list(range(len(self._samples)))  # [0,1,2,3,...]

        self.expand_data(expanding_rate)

    def expand_data(self, rate: int) -> None:
        if rate > 1:
            self._samples_idx_map *= rate

    @staticmethod
    def find_label_path(path: str) -> str:
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        return path.replace(ext, '.json')

    def load_data(self) -> None:
        for image_path in tqdm(self.all_image_path, desc='Loading data'):

            label = MaskLabel()  # Empty label
            image = Image(path=image_path)  # Just only have image path

            label_path: str = self.find_label_path(image_path)
            if os.path.exists(label_path) is True:

                # load and decode json data
                label.set_metadata(load_json(label_path))

                # check image path
                if os.path.basename(image_path) != label.image_path:
                    logger.warning(f'img_path != json.imagePath,{image_path}')
                    continue

                self.samples_with_label.append((image, label))

            else:
                self.background_samples.append((image, label))

        logger.info(f'samples_with_label: {len(self.samples_with_label)}')
        logger.info(f'background_samples: {len(self.background_samples)}')

    def cache_images_to_memory(self) -> None:

        image: Image
        label: MaskLabel
        if len(self.samples_with_label) > 0:
            for image, label in tqdm(self.samples_with_label, desc='Preload Image'):
                im = self._load_image(image.path)
                iw, ih = get_image_wh(im)
                image.data = letterbox(im, self._wh)
                label.mask = self.get_mask(label.objects, (iw, ih))

        if len(self.background_samples) > 0:
            for image, label in tqdm(self.background_samples, desc='Preload Background'):
                im = self._load_image(image.path)
                image.data = letterbox(im, self._wh)
                label.mask = np.zeros((self._hw[0], self._hw[1], 1), dtype=np.uint8)

    def get_mask(self, objects: list, image_wh: Tuple[int, int]) -> np.ndarray:

        iw, ih = image_wh[0], image_wh[1]  # image wh
        ow, oh = self._wh[0], self._wh[1]  # input wh

        if not objects:
            return np.zeros((oh, ow, 1), dtype=np.uint8)

        mask = np.zeros((oh, ow), dtype=np.uint8)
        for obj in objects:
            # points:[[x,y],[x,y],...]

            points = np.array(obj["points"], dtype=float)

            if (iw, ih) != (ow, oh):
                points /= np.array([iw, ih], dtype=float)  # x/iw y/ih
                points *= np.array([ow, oh], dtype=float)  # x*ow y*oh
                points = safe_round(points)

            mask = self.polygon2mask(mask, points, self._labels[obj['label']])

        # (H,W)->(H,W,C) C=1
        mask = hw_to_hw1(mask)

        return mask

    @staticmethod
    def polygon2mask(mask: np.ndarray, points: np.ndarray, label_idx: Optional[int] = 0) -> np.ndarray:
        assert 0 <= label_idx <= 255, '255 >= label_idx >= 0'
        if points.dtype != np.int32:
            points = points.astype(np.int32)
        cv2.fillConvexPoly(mask, points, color=label_idx)  # noqa
        return mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_idx = self._samples_idx_map[index]

        image: Image
        label: MaskLabel
        image, label = self._samples[sample_idx]

        im = image.data if self._use_cache else self._load_image(image.path)
        iw, ih = get_image_wh(im)
        mask = label.mask if self._use_cache else self.get_mask(label.objects, (iw, ih))

        im, mask = self._transform((im, mask))

        return im, mask

    def __len__(self) -> int:
        return len(self._samples_idx_map)
