import os
import os.path as osp
import sys
import json
import random
from abc import ABC
import pandas as pd
import cv2
import numpy as np

from PIL import Image
from PIL import ImageDraw
from typing import Optional, Callable, Any, Tuple, Dict, List, Union
from torch.utils.data import Dataset
from loguru import logger

__all__ = [
    'BaseDataset',
    'ClassificationDataset',
    'SegmentationDataSet',
]


class BaseDataset(Dataset, ABC):
    def __init__(self, loader_type: str = 'pil', img_type: Optional[str] = 'RGB'):
        self.IMG_EXTENSIONS = [
            '.jpg', '.jpeg', '.png',
        ]
        self.loader_type = loader_type
        self.loader = self.get_image_loader(loader_type)

        self._support_img_types = ['RGB', 'GRAY']
        assert img_type in self._support_img_types, logger.error('Img mode is not support.')
        self.img_type = img_type

    @staticmethod
    def expanding_data(samples: list, rate: int = 0):
        assert rate >= 0, f'Expanding data rate<0.'

        new_samples = []
        for i in range(rate):
            new_samples += samples
        random.shuffle(new_samples)
        return new_samples

    @staticmethod
    def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
        return filename.lower().endswith(extensions)

    def is_valid_image(self, filename: str) -> bool:
        return self.has_file_allowed_extension(filename, self.IMG_EXTENSIONS)

    # image loader e.g. PIL or OpenCV or IPP(inter)
    def pil_loader(self, path: str) -> Image.Image:
        img = Image.open(path)

        if self.img_type == 'RGB':
            if img.mode != self.img_type:
                img = img.convert(self.img_type)

        if self.img_type == 'GRAY':
            if img.mode != 'L':
                img = img.convert('L')

        return img

    def get_image_loader(self, loader_type: str) -> Callable:
        if loader_type == 'opencv':
            return self.opencv_loader
        elif loader_type == 'pil':
            return self.pil_loader

    def opencv_loader(self, path: str, rgb2bgr: bool = False) -> np.ndarray:
        im = cv2.imread(path)

        if self.img_type == 'RGB':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.img_type == 'GRAY':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        return im

    @staticmethod
    def save_class_to_id_map(save_path: str, class_to_idx: dict) -> None:
        with open(save_path, 'w') as f:
            # classes_to_idx=classes1:0,classes1:1,...
            # classes_id_map = 0:classes1,1:classes2,...
            f.writelines([f'{idx}:{cls}\n' for cls, idx in class_to_idx.items()])

    @staticmethod
    def load_class_to_id_map(class_to_id_map_path: str) -> dict:

        if not os.path.exists(class_to_id_map_path):
            return {}

        class_id_map = {}
        with open(class_to_id_map_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                # k=0,1,2,...
                # v=classes1,classes2,...
                idx, cls = line.split(":", 1)
                # class_id_map={0:"classes1",1:"classes2",...}
                class_id_map.update({int(idx): cls})
        return class_id_map

    def get_classes(self, path: str) -> list:
        return self.get_dirs(path)

    @staticmethod
    def get_dirs(path: str) -> list:
        classes = []
        for d in os.scandir(path):
            if d.is_dir():
                classes.append(d.name)
        classes.sort()
        return classes

    @staticmethod
    def gen_classes_to_idx_map(classes: list) -> dict:
        # class_to_idx={classes0:0,classes1:1,...}
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return class_to_idx

    @staticmethod
    def gen_idx_to_classes_map(classes: list) -> dict:
        # idx_to_cls={0:cls0,1:cls1,...}
        idx_to_cls = {idx: cls for idx, cls in enumerate(classes)}
        return idx_to_cls

    @staticmethod
    def get_all_file(path: str) -> list:
        all_data = []
        for root, dirs, files in os.walk(path):
            for file in files:
                all_data.append(os.path.join(root, file))
        return all_data

    def get_file_by_subfix(self, path: str, subfix: Union[str, list]) -> list:
        all_data = self.get_all_file(path)
        data = []
        for file in all_data:
            file_basename = os.path.basename(file)
            name, ext = os.path.splitext(file_basename)

            if isinstance(subfix, list):
                if ext in subfix:
                    data.append(file)
            elif isinstance(subfix, str):
                if ext == subfix:
                    data.append(file)
        return data

    @staticmethod
    def letterbox(
            image_src: np.ndarray,
            dst_size: tuple,  # hw
            pad_color: tuple = (114, 114, 114)
    ) -> tuple:

        src_h, src_w = image_src.shape[:2]
        # dst_h, dst_w = dst_size
        dst_w, dst_h = dst_size
        scale = min(dst_h / src_h, dst_w / src_w)
        pad_h, pad_w = int(round(src_h * scale)), int(round(src_w * scale))

        if image_src.shape[0:2] != (pad_w, pad_h):
            image_dst = cv2.resize(
                image_src, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_dst = image_src

        top = int((dst_h - pad_h) / 2)
        down = int((dst_h - pad_h + 1) / 2)
        left = int((dst_w - pad_w) / 2)
        right = int((dst_w - pad_w + 1) / 2)

        # add border
        image_dst = cv2.copyMakeBorder(
            image_dst, top, down, left, right, cv2.BORDER_CONSTANT, value=pad_color)

        x_offset, y_offset = max(left, right) / dst_w, max(top, down) / dst_h
        return image_dst, x_offset, y_offset


class ClassificationDataset(BaseDataset):
    def __init__(
            self,
            root: str,
            wh: Optional[Union[list, tuple]] = None,
            loader_type: str = 'pil',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            expanding_rate: Optional[int] = 0,
            letterbox: Optional[bool] = False,
            img_type: Optional[str] = 'RGB',
    ):
        super(ClassificationDataset, self).__init__(loader_type=loader_type, img_type=img_type)
        self.root = root
        self.wh = wh

        self.letterbox_flag = letterbox
        self.letterbox_color = (114, 114, 114)

        self._labels = self.get_dirs(root)
        self._classes_to_idx = self.gen_classes_to_idx_map(self._labels)

        # self.loader_type = loader_type
        # self.loader = self.get_image_loader(loader_type)

        self.transform = transform
        self.target_transform = target_transform

        self.samples = self.get_samples(root, class_to_idx=self._classes_to_idx)

        if expanding_rate != 0:
            self.samples += self.expanding_data(self.samples, expanding_rate)

        self.targets = [s[1] for s in self.samples]
        if len(self.samples) == 0:
            logger.warning(f"Found 0 files in sub folders of: {self.root}\n")

    @property
    def get_labels(self) -> list:
        return self._labels

    @property
    def get_classes_to_idx(self) -> dict:
        return self._classes_to_idx

    def get_samples(self, path: str, class_to_idx: dict) -> list:
        # class_to_idx = kwargs.get('class_to_idx')

        samples = []
        path = os.path.expanduser(path)

        # target:classes0,classes1,...(low->high)
        for classes in sorted(class_to_idx.keys()):
            target_path = os.path.join(path, classes)

            if not os.path.exists(target_path):
                logger.warning(f'Don`t not found path:{target_path}')
                continue

            for root, dirs, files in os.walk(target_path):
                for file in files:
                    if self.is_valid_image(file):
                        file = os.path.join(root, file)
                        item = (file, class_to_idx[classes])
                        # samples=[('xxx/xxx.jpg',1),(xxx/xxx.jpg,0),...]
                        samples.append(item)
        random.shuffle(samples)
        return samples

    def __getitem__(self, index: int) -> Any:
        path, label = self.samples[index]
        image: Union[Image.Image, np.ndarray] = self.loader(path)

        if (self.letterbox_flag is True) and (self.wh is not None):
            sw: int = -1
            sh: int = -1

            if self.loader_type == 'pil':
                sw, sh = image.size
            if self.loader == 'opencv':
                sh, sw = image.shape
            assert sw > 0 and sh > 0, f'Error: sw or sh <0'

            if sh != self.wh[1] or sw != self.wh[0]:
                if isinstance(image, np.ndarray) is False:
                    image = np.asarray(image)  # PIL.Image -> numpy.ndarray

                image, _, _ = self.letterbox(image, dst_size=self.wh, pad_color=self.letterbox_color)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.samples)


class SegmentationDataSet(BaseDataset):
    def __init__(
            self,
            root: str,
            loader_type: str = 'pil',
            add_background: bool = True,
            transform: Optional[Callable] = None,  # to samples
            target_transform: Optional[Callable] = None,  # to target
            is_training: Optional[bool] = False,
            expanding_rate: Optional[int] = 0,
            img_type: Optional[str] = 'RGB',
    ) -> None:
        super(SegmentationDataSet, self).__init__(loader_type=loader_type, img_type=img_type)
        self.root = root
        self.is_training = is_training
        self.train_transform = transform
        self.target_transform = target_transform

        self._labels = self.find_seg_classes(self.root, add_background)
        self._classes_to_idx = self.gen_classes_to_idx_map(self._labels)
        # self.loader_type = loader_type
        # self.loader = self.get_image_loader(loader_type)
        self.samples = self.get_file_by_subfix(self.root, self.IMG_EXTENSIONS)

        if expanding_rate != 0:
            self.samples += self.expanding_data(self.samples, expanding_rate)

        self.samples_info, self.samples_with_label, self.background_samples = self.data_prefetch(self.samples)
        self.samples = self.samples_with_label + self.background_samples

    @property
    def labels(self) -> list:
        return self._labels

    @property
    def class_to_idx(self) -> dict:
        return self._classes_to_idx

    # check json file
    @staticmethod
    def json_loader(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def find_seg_classes(self, root: str, add_background: bool = True) -> list:

        classes = ['0_background_'] if add_background else []
        json_files = self.get_file_by_subfix(root, '.json')
        for json_item in json_files:
            if os.path.exists(json_item):
                key_points_json = self.json_loader(json_item)
                # "shapes": [
                #     {
                #         "label": "qipao",
                #         "points": []
                #       }
                # ]
                for key_points in key_points_json.get("shapes"):
                    if key_points["label"] not in classes:
                        classes.append(key_points["label"])

        return sorted(classes)

    def data_prefetch(self, samples: list) -> tuple:
        samples_with_label = []
        background_samples = []
        samples_info = {}

        for img_path in samples:

            img_basename = os.path.basename(img_path)
            name, ext = os.path.splitext(img_basename)
            label_path = img_path.replace(ext, '.json')

            if os.path.exists(label_path):
                label_data = self.json_loader(label_path)

                # 首先要进行关键点按label进行排序，因为labelme存储的标签可能不是从label开始的。
                sort_res = sorted(label_data.get('shapes'),
                                  key=lambda x: x.get('label'))

                key_points_data = []  # [[(x,y),...],[(x,y),...]]
                key_points_cls = []  # [cls1,cls2,...]
                for key_point in sort_res:
                    if key_point.get("shape_type") == "polygon":
                        polygon_key_points = key_point["points"]
                    else:  # rectangle
                        point1 = key_point["points"][0]
                        point2 = [key_point["points"][1][0], key_point["points"][0][1]]
                        point3 = key_point["points"][1]
                        point4 = [key_point["points"][0][0], key_point["points"][1][1]]
                        polygon_key_points = [point1, point2, point3, point4]

                    key_points_data.append(polygon_key_points)

                    cur_idx = self._classes_to_idx.get(key_point["label"])

                    key_points_cls.append(cur_idx)

                if label_data.get("imagePath") != img_basename:
                    logger.warning(f"json file->imagePath != image name")
                    continue

                samples_info.update({
                    img_path: [key_points_data, key_points_cls]
                })
                samples_with_label.append(img_path)

            else:
                background_samples.append(img_path)

        return samples_info, samples_with_label, background_samples

    @staticmethod
    def landmark_to_mask_vec(mask: np.ndarray, key_points_list: list, class_id: int = 1) -> np.ndarray:
        mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask)
        xy = [tuple(point) for point in key_points_list]
        assert len(xy) != 1
        draw.polygon(xy=xy, outline=0, fill=class_id)
        mask = np.array(mask, dtype=np.uint8)
        return mask

    def __getitem__(self, index: int):
        path = self.samples[index]
        image: Union[Image.Image, np.ndarray] = self.loader(path)

        curr_sample_info = self.samples_info.get(path)
        iw, ih = 0, 0
        if self.loader_type == 'pil':
            iw, ih = image.size
        if self.loader_type == 'opencv':
            ih, iw = image.shape
        assert iw > 0 and ih > 0, f'iw<=0 or ih<=0'

        # info is None == this is background image
        if curr_sample_info is None:
            landmark = np.zeros((ih, iw), dtype=np.uint8)

        else:
            landmark, keypoint_cls = curr_sample_info

            if not landmark:  # landmark==[]
                landmark = np.zeros((ih, iw), dtype=np.uint8)
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

        if self.train_transform is not None:
            image = self.train_transform(image)
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
        return len(self.samples)
