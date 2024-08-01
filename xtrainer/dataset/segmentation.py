import os
from typing import Optional, Callable, List, Union, Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from xtrainer.dataset import Image, SegLabel
from xtrainer.dataset.base import BaseDataset
from xtrainer.utils.common import (
    load_json,
    get_images,
    get_image_shape,
    hw_to_hw1,
    safe_round
)


class SegmentationDataSet(BaseDataset):
    def __init__(
        self,
        root: str,
        wh: Tuple[int, int],
        loader_type: Optional[str] = 'opencv',
        img_type: Optional[str] = 'RGB',
        transform: Optional[Callable] = None,  # to samples
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

    @staticmethod
    def find_label_path(path: str) -> str:
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        return path.replace(ext, '.json')

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
            image.data = self._load_image(image.path)
            ih, iw = get_image_shape(image)
            label.mask = self.get_mask(label.objects, (ih, iw))

        for image, label in tqdm(self.background_samples):
            image.data = self._load_image(image.path)
            label.mask = np.zeros((self._hw[0], self._hw[1], 1), dtype=np.uint8)

    # Return mask shape=(input_h,input_w,1,np.uint8)
    def get_mask(self, objects: list, image_hw: Tuple[int, int]) -> np.ndarray:

        ih, iw = image_hw[0], image_hw[0]  # image hw
        oh, ow = self._wh[1], self._wh[1]  # input hw

        if not objects:
            return np.zeros((oh, ow, 1), dtype=np.uint8)

        mask = np.zeros((oh, ow), dtype=np.uint8)
        for obj in objects:
            # points:[[x,y],[x,y],...]

            points = np.array(obj["points"], dtype=float)
            points /= np.array([iw, ih], dtype=float)  # x/iw y/ih
            points *= np.array([ow, oh], dtype=float)  # x*ow y*oh
            points = safe_round(points)

            label: str = obj['label']
            mask = self.polygon2mask(mask, points, self.label2idx(label))

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
        sample_idx = self._samples_map[index]

        image: Image
        label: SegLabel
        image, label = self._samples[sample_idx]

        im = image.data if self._is_preload else self._load_image(image.path)
        h, w = get_image_shape(im)
        mask = label.mask if self._is_preload else self.get_mask(label.objects, (h, w))
        img_h, img_w = get_image_shape(im)
        assert img_w > 0 and img_h > 0, 'Error: img_w or img_h <=0'

        # input.type=[Image.Image,torch.Tensor]
        data = (im, mask)
        im, mask = self._transform(data)

        return im, mask

    def __len__(self) -> int:
        return len(self._samples)


if __name__ == '__main__':
    from xtrainer.core.preprocess import SegImageT
    from torch.utils.data import DataLoader
    from time import time
    import cv2

    wh = (224, 224)
    ds = SegmentationDataSet(
        root=r'C:\Users\Lee Linfeng\Desktop\temp\20240518-bad-F',
        wh=wh,
        is_preload=True,
        transform=SegImageT(wh),
    )
    # image, mask = ds[0]
    # mask *= 100
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # print(image.shape)
    # print(mask.shape)
    # cv2.imwrite('im.jpg', image)
    # cv2.imwrite('mask.jpg', mask)

    t1 = time()
    dl = DataLoader(ds, batch_size=8)
    for imgs, target in dl:
        print(imgs.shape)
        print(target.shape)

    t2 = time()
    print(t2 - t1)
