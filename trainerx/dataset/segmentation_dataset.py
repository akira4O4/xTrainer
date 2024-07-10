import os
import random
from copy import deepcopy
from typing import Optional, Callable, List, Union, Tuple

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from loguru import logger
from tqdm import tqdm

from trainerx.dataset.base_dataset import BaseDataset
from trainerx.utils.common import (
    get_json_file,
    load_json,
    get_images,
    check_size,
    get_image_shape,
    exchange_wh
)
from trainerx.core.transforms import LetterBox


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
        load_all_data: Optional[bool] = False
    ) -> None:
        super(SegmentationDataSet, self).__init__(
            root=root,
            wh=wh,
            loader_type=loader_type,
            img_type=img_type,
            transform=transform,
            target_transform=target_transform,
            load_all_data=load_all_data
        )
        self.letterbox = LetterBox(exchange_wh(self._wh))
        self.background_mask = np.zeros((self._wh[1], self._wh[0]), dtype=np.uint8)

        # self.is_training = is_training

        self._labels = ['0_background_']

        self.find_labels()
        self.load_data()

        # samples=[
        # (img_path,label_path),
        # (img_path,label_path),
        # ]
        self.samples_with_label: List[Tuple[str, dict]] = []
        self.background_samples: List[Tuple[str, dict]] = []
        self._samples = self.samples_with_label + self.background_samples

        if expanding_rate != 0:
            self.expanding_data(expanding_rate)

    def find_labels(self) -> None:

        json_files: List[str] = get_json_file(self._root)

        logger.info(f'finding label...')
        for json_item in tqdm(json_files):
            if os.path.exists(json_item):
                key_points_json = load_json(json_item)
                '''
                shapes": [
                   {
                       "label": "",
                       "points": []
                    }
                ]
                '''
                for shape in key_points_json.get("shapes"):
                    if shape["label"] not in self._labels:
                        self._labels.append(shape["label"])
        self._labels.sort()

    @staticmethod
    # find label path from img
    def img2label_path(path: str) -> str:
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        label_path = path.replace(ext, '.json')
        return label_path

    def load_data(self) -> None:
        images: List[str] = get_images(self._root, self._SUPPORT_IMG_FORMAT)

        if self._load_all_data:
            self.load_all_data_to_memory(images)
        else:
            image: str
            for image in tqdm(images):

                label_path: str = self.img2label_path(image)

                if os.path.exists(label_path):
                    data = load_json(label_path)
                    # check image path
                    if image != data['imagePath']:
                        logger.warning(f'img_path != json.imagePath,{image}')
                        continue
                    self.samples_with_label.append((image, data))
                else:
                    self.background_samples.append((image, {}))

        logger.info(f'samples_with_label: {len(self.samples_with_label)}')
        logger.info(f'background_samples: {len(self.background_samples)}')

    def load_all_data_to_memory(self, image_path: List[str]) -> None:
        # self.background_samples:List[np.ndarray]
        # self.samples_with_label:List[dict]

        for path in tqdm(image_path):
            image: Union[Image.Image, np.ndarray] = self._image_loader(path)

            label_path = self.img2label_path(path)

            if os.path.exists(label_path):
                self.background_samples.append(image)
            else:
                self.samples_with_label.append({
                    path: image,
                    label_path: load_json(label_path)
                })

        random.shuffle(self.background_samples)
        random.shuffle(self.samples_with_label)

    def get_mask(self, data: dict) -> np.ndarray:

        # shapes:[{label,points},{},...]
        shapes: list = data.get('shapes')

        # 首先要进行关键点按label进行排序，因为labelme存储的标签可能不是从label开始的。
        shapes = sorted(shapes, key=lambda x: x.get('label'))

        all_points = []  # [[(x,y),...],[(x,y),...]]
        all_label_idx = []  # [cls1,cls2,...]

        # points:[[x,y],[x,y],...]
        shape: dict
        for shape in shapes:
            points: list = shape["points"]
            all_points.append(points)
            all_label_idx.append(self._labels.index(shape["label"]))

    def polygon2mask(self, polygons: list, label_idx: Optional[int] = 0, ) -> np.ndarray:

        # polygon=[[x,y],[x,y],...]
        # polygons=[polygon1,polygon2,...]

        assert label_idx >= 0, 'label_idx<0'

        mask = deepcopy(self.background_mask)
        polygons = np.asarray(polygons, dtype=np.int32)

        # polygons.shape=[N,rows,2]
        polygons = polygons.reshape((polygons.shape[0], -1, 2))
        cv2.fillPoly(mask, polygons, color=label_idx)
        return mask

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:

        img_path, label_data = self._samples[index]

        image: Union[Image.Image, np.ndarray] = self._loader(img_path)

        # check image
        img_w, img_h = get_image_shape(image)

        mask: np.ndarray = self.background_mask

        if not check_size(image, self._wh):
            image = self.letterbox(image)

        if label_data != {}:
            mask = self.get_mask(label_data)

            if self._transform is not None:
                image = self._transform(image)

            if self._target_transform is not None:
                mask = self._target_transform(mask)

        mask = mask[None]  # (h, w) -> (1, h, w)
        return image, mask

    def __len__(self) -> int:
        return len(self._samples)


def polygon2mask(
    imgsz: tuple,
    polygons: List[float],
    color=1,
    downsample_ratio=1
):
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    return cv2.resize(mask, (nw, nh))


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        t = segments[si].reshape(-1)
        mask = polygon2mask(imgsz, [t], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    a = -areas
    index = np.argsort(-areas)

    ms = np.array(ms)
    ms = ms[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


if __name__ == '__main__':
    data = np.array([[1, 1],
                     [8, 8],
                     [2, 7]])
    temp = data[None]
    # temp[0][0] = 100
    print(temp)
    print(data)
    # polygons2masks_overlap(
    #     (10, 10),
    #     np.array([
    #         [[2, 2],
    #          [8, 8],
    #          [2, 7]],
    #
    #         [[1, 1],
    #          [8, 8],
    #          [2, 7]],
    #     ])
    # )
