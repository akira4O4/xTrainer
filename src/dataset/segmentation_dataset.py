import os
import random
from typing import Optional, Callable, List, Union

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from loguru import logger
from tqdm import tqdm
from ..utils.common import get_json_file, load_json, get_images
from base_dataset import BaseDataset


class SegmentationDataSet(BaseDataset):
    def __init__(
        self,
        root: str,
        wh: Optional[list] = None,
        loader_type: str = 'pil',
        img_type: Optional[str] = 'RGB',
        transform: Optional[Callable] = None,  # to samples
        target_transform: Optional[Callable] = None,  # to target
        expanding_rate: Optional[int] = 0,
        is_training: Optional[bool] = False,
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

        self.is_training = is_training

        self._labels = ['0_background_']

        self.find_labels()
        self.load_data()
        # self._samples = self.get_file_by_subfix()

        if expanding_rate != 0:
            self.expanding_data(expanding_rate)

        self.samples_with_label = []
        self.background_samples = []
        self.all_samples = []

        self.data_prefetch()
        self._samples = self.samples_with_label + self.background_samples

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
    def convert_to_json(path: str) -> str:
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        label_path = path.replace(ext, '.json')
        return label_path

    def load_data(self) -> None:
        images = get_images(self._root, self._DEFAULT_SUPPORT_IMG_SUFFIX)
        # random.shuffle(self._samples)

        if self._load_all_data:
            self.load_all_data_to_memory(images)
        else:
            for image in images:
                self.samples_with_label = images

                label_path = self.convert_to_json(image)

                if os.path.exists(label_path):
                    # self.background_samples:List[str]
                    self.background_samples.append(label_path)
                else:
                    ...

    def load_all_data_to_memory(self, image_path: List[str]) -> None:
        # self.background_samples:List[np.ndarray]
        # self.samples_with_label:List[dict]

        for path in tqdm(image_path):
            image: Union[Image.Image, np.ndarray] = self._image_loader(path)

            label_path = self.convert_to_json(path)

            if os.path.exists(label_path):
                self.background_samples.append(image)
            else:
                self.samples_with_label.append({
                    path: image,
                    label_path: load_json(label_path)
                })

        random.shuffle(self.background_samples)
        random.shuffle(self.samples_with_label)

    def point_prefetch(self, data: dict):
        # shapes:[{label,points},{},...]
        shapes = data.get('shapes')

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

        self.samples_info.update({
            img_path: [total_points, total_label_idx]
        })
        self.samples_with_label.append(img_path)

    def data_prefetch(self):
        for img_path in self._samples:

            img_basename = os.path.basename(img_path)
            name, ext = os.path.splitext(img_basename)
            label_path = img_path.replace(ext, '.json')

            if os.path.exists(label_path):
                label_data = load_json(label_path)

                # shapes:[{label,points},{},...]
                shapes = label_data.get('shapes')

                # 首先要进行关键点按label进行排序，因为labelme存储的标签可能不是从label开始的。
                shapes = sorted(shapes, key=lambda x: x.get('label'))

                total_points = []  # [[(x,y),...],[(x,y),...]]
                total_label_idx = []  # [cls1,cls2,...]

                # points:[[x,y],[x,y],...]
                points: List[List[float]] = []
                for shape in shapes:

                    if shape.get("shape_type") == "polygon":
                        points = shape["points"]

                    elif shape.get("shape_type") == "rectangle":  # rectangle
                        point1 = shape["points"][0]
                        point2 = [shape["points"][1][0], shape["points"][0][1]]
                        point3 = shape["points"][1]
                        point4 = [shape["points"][0][0], shape["points"][1][1]]
                        points = [point1, point2, point3, point4]

                    total_points.append(points)

                    label_idx: int = self._labels.index(shape["label"])

                    total_label_idx.append(label_idx)

                if label_data.get("imagePath") != img_basename:
                    logger.warning(f"json file->imagePath != image name")
                    continue

                self.samples_info.update({
                    img_path: [total_points, total_label_idx]
                })
                self.samples_with_label.append(img_path)

            else:
                self.background_samples.append(img_path)

    @staticmethod
    def landmark_to_mask_vec(
        mask: np.ndarray,
        key_points_list: list,
        class_id: int = 1
    ) -> np.ndarray:

        mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask)
        xy = [tuple(point) for point in key_points_list]
        assert len(xy) != 1
        draw.polygon(xy=xy, outline=0, fill=class_id)
        mask = np.array(mask, dtype=np.uint8)
        return mask

    def __getitem__(self, index: int):
        path = self._samples[index]
        image: Union[Image.Image, np.ndarray] = self._image_loader(path)

        curr_sample_info = self.samples_info.get(path)
        img_w, img_h = 0, 0
        if isinstance(image, Image.Image):
            img_w, img_h = image.size
        elif isinstance(image, np.ndarray):
            img_h, img_w = image.shape
        assert img_w > 0 and img_h > 0, 'img w<=0 or img h<=0'

        # info is None == this is background image
        if curr_sample_info is None:
            landmark = np.zeros((img_h, img_w), dtype=np.uint8)

        else:
            landmark, keypoint_cls = curr_sample_info

            if not landmark:  # landmark==[]
                landmark = np.zeros((img_h, img_w), dtype=np.uint8)
            else:
                landmark_len = list(map(lambda x: len(x), landmark))

                if len(landmark_len) >= 1:
                    new_landmark = []
                    for i in range(len(landmark_len)):
                        new_landmark.extend(landmark[i])
                    landmark = new_landmark.copy()

                landmark = np.asarray(landmark, dtype=np.float32)

                if self.target_transform is not None:

                    if self.is_training:
                        # image type:np.ndarray
                        image, landmark = self.target_transform(image, landmark)
                    # image.size=wh
                    # mask.shape=hw
                    mask = np.zeros((image.height, image.width), dtype=np.uint8)
                    start_len = 0

                    for idx, landmark_len_item in enumerate(landmark_len):
                        cur_landmark = landmark[start_len:(start_len + landmark_len_item)]
                        start_len += landmark_len_item
                        cur_keypoint_class = keypoint_cls[idx]
                        mask = self.landmark_to_mask_vec(mask, cur_landmark, cur_keypoint_class)
                    landmark = mask.copy()

                    # val mode
                    if self.is_training is False:
                        #   image.h!=landmark.h or image.w!=landmark.w
                        if image.height != landmark.shape[0] or image.width != landmark.shape[1]:
                            # cv2.resize() params:dst.shape=(w,h)
                            landmark = cv2.resize(landmark, (image.width, image.height))

        if self.transform is not None:
            image = self.transform(image)
            # image.shape=[3,h,w]
            # landmark.shape=[h,w]
            if image.shape[1] != landmark.shape[0] or image.shape[2] != landmark.shape[1]:
                # cv2.resize::dsize=(w,h)
                landmark = cv2.resize(landmark, (image.shape[2], image.shape[1]))

        # landmark = np.array(landmark, dtype=np.float32)
        # shape=[1,h,w]
        landmark = landmark[np.newaxis, :, :]

        # return image, target, landmark
        return image, landmark

    def __len__(self) -> int:
        return len(self._samples)
