import os
import random
from abc import ABC
from typing import Optional, Callable, Tuple, List, Union

import cv2
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
from loguru import logger
from torch.utils.data import Dataset

from .utils import get_images, load_json

__all__ = [
    'BaseDataset',
    'ClassificationDataset',
    'SegmentationDataSet',
]


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        root: str,
        loader_type: str = 'pil',
        img_type: Optional[str] = 'RGB',
        transform: Optional[Callable] = None,  # to samples
        target_transform: Optional[Callable] = None,  # to target
    ) -> None:

        self.root = root
        self.support_img_suffix = ('.jpg', '.jpeg', '.png')
        self.support_img_type = ['RGB', 'GRAY']

        self.loader_type = loader_type
        self.image_loader = self.get_image_loader(loader_type)

        self.transform = transform
        self.target_transform = target_transform

        self.samples = []

        self._labels: List[str] = []

        assert img_type in self.support_img_type, 'Image type is not support.'
        self.img_type = img_type

    @property
    def labels(self) -> list:
        return self._labels

    @property
    def num_of_label(self) -> int:
        return len(self._labels)

    @property
    def data_size(self) -> int:
        return len(self.samples)

    def set_transform(self, val) -> None:
        self.transform = val

    def set_target_transform(self, val) -> None:
        self.target_transform = val

    def expanding_data(self, rate: int = 0):
        if rate == 0:
            return

        new_data = []
        for i in range(rate):
            new_data += self.samples
        random.shuffle(new_data)
        self.samples = new_data

    def check_image_suffix(self, filename: str) -> bool:
        return filename.lower().endswith(self.support_img_suffix)

    def pil_loader(self, path: str) -> Image.Image:
        img = Image.open(path)

        if self.img_type == 'RGB':
            if img.mode != self.img_type:
                img = img.convert(self.img_type)

        if self.img_type == 'GRAY':
            if img.mode != 'L':
                img = img.convert('L')

        return img

    def opencv_loader(self, path: str) -> np.ndarray:
        im = cv2.imread(path)

        if self.img_type == 'RGB':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.img_type == 'GRAY':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        return im

    def get_image_loader(self, loader_type: str) -> Callable:
        if loader_type == 'opencv':
            return self.opencv_loader
        elif loader_type == 'pil':
            return self.pil_loader

    @staticmethod
    def get_all_file(path: str) -> list:
        all_data = []
        for root, dirs, files in os.walk(path):
            for file in files:
                all_data.append(os.path.join(root, file))
        return all_data

    def get_file_by_subfix(self) -> list:
        all_data = self.get_all_file(self.root)
        data = []
        for file in all_data:
            file_basename = os.path.basename(file)
            name, subfix = os.path.splitext(file_basename)
            if subfix in self.support_img_suffix:
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
        wh: Optional[list] = None,
        loader_type: str = 'pil',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        expanding_rate: Optional[int] = 0,
        letterbox: Optional[bool] = False,
        img_type: Optional[str] = 'RGB',
    ):
        super(ClassificationDataset, self).__init__(
            root=root,
            loader_type=loader_type,
            img_type=img_type,
            transform=transform,
            target_transform=target_transform
        )

        self.root = root
        self.wh = wh

        self.letterbox_flag = letterbox
        self.letterbox_color = (114, 114, 114)

        self.find_labels()

        # samples=[('xxx/xxx.jpg',1),(xxx/xxx.jpg,0),...]
        self.samples: List[Tuple[str, int]] = []
        self.load_data()

        if expanding_rate > 0:
            self.expanding_data(expanding_rate)

        self.targets = [s[1] for s in self.samples]
        if len(self.samples) == 0:
            logger.warning(f"Found 0 files in sub folders of: {self.root}\n")

    def find_labels(self) -> None:
        for d in os.scandir(self.root):
            if d.is_dir():
                self._labels.append(d.name)
        self._labels.sort()

    def load_data(self) -> None:

        # samples=[('xxx/xxx.jpg',1),(xxx/xxx.jpg,0),...]

        for idx in range(self.num_of_label):
            target_path = os.path.join(self.root, self._labels[idx])

            images: List[str] = get_images(target_path, self.support_img_suffix)

            # print(f'{self.labels[idx]}: {len(images)}')
            self.samples.extend(list(map(lambda x: (x, idx), images)))  # noqa

        random.shuffle(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image: Union[Image.Image, np.ndarray] = self.image_loader(path)

        if (self.letterbox_flag is True) and (self.wh is not None):
            sw: int = -1
            sh: int = -1

            if self.loader_type == 'pil':
                sw, sh = image.size
            elif self.loader_type == 'opencv':
                sh, sw = image.shape

            assert sw > 0 and sh > 0, f'Error: sw or sh <0'

            if sh != self.wh[1] or sw != self.wh[0]:
                if isinstance(image, np.ndarray) is False:
                    # PIL.Image -> numpy.ndarray
                    image = np.asarray(image)  # noqa

                image, _, _ = self.letterbox(image, dst_size=tuple(self.wh), pad_color=self.letterbox_color)

        image: torch.Tensor
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
        super(SegmentationDataSet, self).__init__(
            root=root,
            loader_type=loader_type,
            img_type=img_type,
            transform=transform,
            target_transform=target_transform
        )
        self.root = root
        self.add_background = add_background
        self.is_training = is_training

        self.find_labels()
        self.samples = self.get_file_by_subfix()

        if expanding_rate != 0:
            self.expanding_data(expanding_rate)

        self.samples_with_label = []
        self.background_samples = []
        self.samples_info = {}

        self.data_prefetch()
        self.samples = self.samples_with_label + self.background_samples

    def find_labels(self):

        self._labels = ['0_background_'] if self.add_background else []
        json_files = self.get_file_by_subfix(self.root, '.json')
        for json_item in json_files:
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

    def data_prefetch(self):
        for img_path in self.samples:

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
        path = self.samples[index]
        image: Union[Image.Image, np.ndarray] = self.image_loader(path)

        curr_sample_info = self.samples_info.get(path)
        iw, ih = 0, 0
        if self.loader_type == 'pil':
            iw, ih = image.size
        elif self.loader_type == 'opencv':
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
        return len(self.samples)
