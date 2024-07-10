import os
from typing import Optional, Callable, List, Union, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger
from tqdm import tqdm

from trainerx.dataset.base_dataset import BaseDataset
from trainerx.utils.common import (
    load_json,
    get_images,
    check_size,
    exchange_wh,
    pil_to_np
)

'''
Labelme Data

{
    "shapes": [
        {
           "label": "",
           "points": [[x,y],[x,y],...]
        },
        {...},
        ...
    ],
    "imagePath": ""
}
'''


class LabelmeData:
    def __init__(self, metadata: Optional[dict] = None) -> None:
        if metadata is None:
            metadata = {}

        self.objects: list = metadata.get('shapes', [])
        self.image_path: str = metadata.get('imagePath', '')
        self.num_of_objects = len(self.objects)
        self.is_background = True if self.num_of_objects == 0 else False
        self.mask = None  # reload mask


class SegmentationDataSet(BaseDataset):
    def __init__(
        self,
        root: str,
        wh: Optional[Tuple[int, int]] = None,
        loader_type: str = 'pil',
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

        self.samples_with_label: List[Tuple[str, LabelmeData]] = []
        self.background_samples: List[Tuple[str, LabelmeData]] = []
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
        images: List[str] = get_images(self._root, self._SUPPORT_IMG_FORMAT)

        image: str
        for image in tqdm(images):

            label_path: str = self.find_label_path(image)

            if os.path.exists(label_path):

                data = LabelmeData(metadata=load_json(label_path))

                # check image path
                if image != data.image_path:
                    logger.warning(f'img_path != json.imagePath,{image}')
                    continue

                # find the all label
                for obj in data.objects:
                    if obj["label"] not in self._labels:
                        self._labels.append(obj["label"])

                # preload mask
                # TODO: add preload image
                if self._preload:
                    data.mask = self.get_mask(data.objects)

                self.samples_with_label.append((image, data))
            else:  # background mask
                data = LabelmeData()

                # preload background mask
                # TODO: add preload image
                if self._preload:
                    data.mask = np.zeros((exchange_wh(self._wh)), dtype=np.uint8)

                self.background_samples.append((image, data))

        self._labels.sort()

        logger.info(f'samples_with_label: {len(self.samples_with_label)}')
        logger.info(f'background_samples: {len(self.background_samples)}')

    def get_mask(self, objects: list) -> np.ndarray:

        mask = np.zeros(exchange_wh(self._wh), dtype=np.uint8)

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        img_path, label_data = self._samples[index]

        image: Union[Image.Image, np.ndarray] = self._loader(img_path)
        image: np.ndarray = pil_to_np(image)

        mask: np.ndarray = self.get_mask(label_data.objects) if self._preload else label_data.mask

        if not check_size(image, self._wh):
            image, x_offset, y_offset = self.letterbox(image)

        if self._transform is not None:
            image = self._transform(image)

        if self._target_transform is not None and not label_data.is_background:
            mask = self._target_transform(mask)

        mask = mask[None]  # (h, w) -> (1, h, w)
        return image, mask

    def __len__(self) -> int:
        return len(self._samples)


if __name__ == '__main__':
    ...
