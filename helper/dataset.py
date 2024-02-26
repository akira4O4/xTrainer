import os
import os.path as osp
import sys
import json
import random
from abc import ABC

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
    def __init__(self,img_mode:Optional[str]='RGB'):
        self.IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png',
                               '.ppm', '.bmp', '.pgm', 
                               '.tif', '.tiff', '.webp'
                               )
        self.loader = None
        self.img_mode=img_mode
        
    @staticmethod
    def expanding_data(samples: list, rate: int = 0):
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
    def pil_loader(self,path: str) -> Image.Image:
        img = Image.open(path) 
        # print(img.mode)
        # print(self.img_mode)
        if img.mode!=self.img_mode:
            img=img.convert(self.img_mode)
        
        return img

    @staticmethod
    def opencv_loader(path: str, rgb2bgr: bool = False) -> np.ndarray:
        im = cv2.imread(path)
        if rgb2bgr:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    @staticmethod
    def json_loader(path: str):
        with open(path, 'r', encoding="utf-8") as f:
            keypoints_json = json.load(f)
        return keypoints_json

    def accimage_loader(self, path: str) -> Any:
        import accimage
        try:
            return accimage.Image(path)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return self.pil_loader(path)

    def default_loader(self, path: str, type: str = 'pil') -> Any:
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            return self.accimage_loader(path)
        else:
            if type == 'opencv':
                return self.opencv_loader(path)
            else:
                # print(self.img_mode)
                return self.pil_loader(path)

    def get_nums_of_dir(self, root: str) -> dict:

        dirs = [d for d in os.listdir(
            root) if os.path.isdir(os.path.join(root, d))]

        nums_of_dir = {}

        for item in dirs:
            # for target in sorted(class_to_idx.keys()):
            nums = 0
            dir_path = os.path.join(root, item)

            if not os.path.isdir(dir_path):
                continue

            for dirPath, dirNames, fileNames in os.walk(dir_path):
                for file in fileNames:
                    file_name, ext = os.path.splitext(file)
                    if ext.lower() in self.IMG_EXTENSIONS:
                        nums += 1
            nums_of_dir.update({item: nums})

        return nums_of_dir

    @staticmethod
    def save_class_id_map(save_path: str, class_to_idx: dict) -> None:
        with open(save_path, 'w') as f:
            # classes_to_idx=classes1:0,classes1:1,...
            # classes_id_map = 0xclasses1,1xclasses2,...
            f.writelines([f'{v}x{k}\n' for k, v in class_to_idx.items()])

    @staticmethod
    def load_class_id_map(class_id_map_path: str) -> dict:

        if not os.path.exists(class_id_map_path):
            return {}

        class_id_map = {}
        with open(class_id_map_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                # k=0,1,2,...
                # v=classes1,classes2,...
                k, v = line.split("x", 1)
                # class_id_map={0:"classes1",1:"classes2",...}
                class_id_map.update({int(k): v})
        return class_id_map

    @staticmethod
    def get_classes(path: str) -> list:
        # classes=[classes1,classes2,...]
        classes = []
        for d in os.scandir(path):
            if d.is_dir():
                classes.append(d.name)
        classes.sort()
        return classes

    @staticmethod
    def get_dirs(path: str) -> list:
        classes = []
        for d in os.scandir(path):
            if d.is_dir():
                classes.append(d.name)
        classes.sort()
        return classes

    @staticmethod
    def classes2idx(classes: list) -> dict:
        # class_to_idx={classes1:0,classes:1,...}
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

    def get_file_with_filter(self, path: str, file_ext: str) -> list:
        all_data = self.get_all_file(path)
        new_data = []
        for file in all_data:
            file_basename = os.path.basename(file)
            if file_ext in file_basename:
                new_data.append(file)
        return new_data

    def get_images(self, path: str) -> list:
        ret = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_name, ext = os.path.splitext(file)
                if ext in self.IMG_EXTENSIONS:
                    ret.append(os.path.join(root, file))
        random.shuffle(ret)
        return ret

    @staticmethod
    def get_all_file(path: str) -> list:
        all_data = []
        for root, dirs, files in os.walk(path):
            for file in files:
                all_data.append(os.path.join(root, file))
        return all_data

    @staticmethod
    def letterbox(
            image_src: np.ndarray,
            dst_size: tuple, #hw
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


class HDF5DataSet:
    ...


class LMDBDataSet:
    ...


class TFRecorderDataSet:
    ...


class ClassificationDataset(BaseDataset):
    def __init__(
            self,
            root: str,
            wh: Optional[Union[list, tuple]] = None,
            loader: Callable = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            expanding_rate: Optional[int] = 0,
            letterbox: Optional[bool] = False,
            img_mode:Optional[str] = 'RGB',
            **kwargs
    ):
        super(ClassificationDataset, self).__init__(img_mode=img_mode)
        self.root = root
        self.wh = wh

        self.letterbox_flag = letterbox
        self.letterbox_color = (114, 114, 114)

        # self.classes = self.get_dirs(root)
        self._labels = self.get_dirs(root)
        self._class_to_idx = self.classes2idx(self._labels)
        self.loader = self.default_loader if loader is None else loader

        self.transform = transform
        self.target_transform = target_transform
        self.samples = self.get_samples(root, class_to_idx=self._class_to_idx)

        if expanding_rate != 0:
            self.samples += self.expanding_data(self.samples, expanding_rate)

        self.targets = [s[1] for s in self.samples]
        if len(self.samples) == 0:
            logger.warning(f"Found 0 files in sub folders of: {self.root}\n")

    @property
    def labels(self) -> list:
        return self._labels

    @property
    def class_to_idx(self) -> dict:
        return self._class_to_idx

    def __getitem__(self, index: int) -> Any:
        path, target = self.samples[index]
        sample = self.loader(path)

        if (self.letterbox_flag is True) and (self.wh is not None):
            sw, sh = sample.size
            # print(sw,sh)
            # print(self.wh)
            if sh != self.wh[1] or sw != self.wh[0]:
                if isinstance(sample, np.ndarray) is False:
                    sample = np.asarray(sample)  # PIL.Image -> numpy.ndarray

                sample, _, _ = self.letterbox(sample,
                                              dst_size=self.wh,
                                              pad_color=self.letterbox_color
                                              )

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def get_samples(self, path: str, class_to_idx: dict) -> list:
        # class_to_idx = kwargs.get('class_to_idx')

        samples = []
        path = os.path.expanduser(path)

        # target:classes0,classes1,...(low->high)
        for target in sorted(class_to_idx.keys()):
            target_path = os.path.join(path, target)

            if not os.path.exists(target_path):
                logger.warning(f'Don`t not found path:{target_path}')
                continue

            for root, dirs, files in os.walk(target_path):
                for file in files:
                    if self.is_valid_image(file):
                        file = os.path.join(root, file)
                        item = (file, class_to_idx[target])
                        # samples=[('xxx/xxx.jpg',1),(xxx/xxx.jpg,0),...]
                        samples.append(item)
        random.shuffle(samples)
        return samples


class SegmentationDataSet(BaseDataset):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = None,
            transform: Optional[Callable] = None,  # to samples
            target_transform: Optional[Callable] = None,  # to target
            is_training: Optional[bool] = False,
            expanding_rate: Optional[int] = 0,
            img_mode:Optional[str] = 'RGB',
            **kwargs
    ) -> None:
        super(SegmentationDataSet, self).__init__(img_mode=img_mode)
        self.root = root
        self.train_transform = transform
        self.target_transform = target_transform
        self.is_training = is_training
        # self.classes = self.find_seg_classes(self.root, background=True)
        self._labels = self.find_seg_classes(self.root, background=True)
        self._class_to_idx = self.classes2idx(self._labels)

        self.loader = self.default_loader if loader is None else loader
        self.samples = self.get_images(self.root)

        if expanding_rate != 0:
            self.samples += self.expanding_data(self.samples, expanding_rate)

        self.samples_info, self.samples_with_label, self.background_samples = self.data_prefetch(
            self.samples)
        self.samples = self.samples_with_label + self.background_samples

    @property
    def labels(self) -> list:
        return self._labels

    @property
    def class_to_idx(self) -> dict:
        return self._class_to_idx
    # check json file

    def find_seg_classes(self, root: str, background: bool = True) -> list:

        classes = ['0_background_'] if background else []
        json_files = self.get_file_with_filter(root, 'json')
        for json_item in json_files:
            if os.path.exists(json_item):
                keypoints_json = self.json_loader(json_item)
                # "shapes": [
                #     {
                #         "label": "qipao",
                #         "points": []
                #       }
                # ]
                for keypoints in keypoints_json.get("shapes"):
                    if keypoints["label"] not in classes:
                        classes.append(keypoints["label"])

        return sorted(classes)

    def data_prefetch(self, samples: list) -> tuple:
        samples_with_label = []
        background_samples = []
        samples_info = {}

        for img_path in samples:
            # img_path = img_info[0]
            # img_cls = img_info[1]
            basename=os.path.basename(img_path)
            name,ext=os.path.splitext(basename)
            label_path = img_path.replace(ext, '.json')

            if os.path.exists(label_path):
                cur_img_name = os.path.basename(img_path)
                label_data = self.json_loader(label_path)

                # 首先要进行关键点按label进行排序，因为labelme存储的标签可能不是从label开始的。
                sort_res = sorted(label_data.get('shapes'),
                                  key=lambda x: x.get('label'))

                keypoints_data = []  # [[(x,y),...],[(x,y),...]]
                keypoints_cls = []  # [cls1,cls2,...]
                for keypoint in sort_res:
                    if keypoint.get("shape_type") == "polygon":
                        polygon_keypoints = keypoint["points"]
                    else:  # rectangle
                        point1 = keypoint["points"][0]
                        point2 = [keypoint["points"][1]
                                  [0], keypoint["points"][0][1]]
                        point3 = keypoint["points"][1]
                        point4 = [keypoint["points"][0]
                                  [0], keypoint["points"][1][1]]
                        polygon_keypoints = [point1, point2, point3, point4]

                    keypoints_data.append(polygon_keypoints)

                    cur_idx = self._class_to_idx.get(keypoint["label"])

                    if cur_idx is None:
                        logger.warning(
                            f"json标签文件里的label跟文件夹命名的label名称不一致,文件{img_path}。")
                        raise
                    else:
                        keypoints_cls.append(cur_idx)

                if label_data.get("imagePath") != cur_img_name:
                    logger.warning(
                        f"标签文件：{cur_img_name}出现的标签里的imagepath跟图片名称不一致。无法解析label数据。")
                    continue
                samples_info.update({
                    img_path: [keypoints_data, keypoints_cls]
                })
                samples_with_label.append(img_path)
            else:  # no json file==background image
                background_samples.append(img_path)

        return samples_info, samples_with_label, background_samples

    def __getitem__(self, index: int):
        # path, target = self.samples[index]
        path = self.samples[index]

        try:
            sample = self.loader(path)
        except:
            logger.warning('异常')
            logger.info(path)
            index0 = np.random.randint(0, len(self.samples) - 1)
            path = self.samples[index0]
            sample = self.loader(path)

        curr_sample_info = self.samples_info.get(path)

        if curr_sample_info is not None:
            landmark, keypoint_cls = curr_sample_info

            if landmark is not None:
                landmark_len = list(map(lambda x: len(x), landmark))

                if len(landmark_len) >= 1:
                    new_landmark = []
                    for i in range(len(landmark_len)):
                        new_landmark.extend(landmark[i])
                    landmark = new_landmark.copy()

                landmark = np.asarray(landmark, dtype=np.float32)
                if self.target_transform is not None:

                    if self.is_training:
                        sample, landmark = self.target_transform(
                            sample, landmark)
                    # sample.size=wh
                    # mask.shape=hw
                    mask = np.zeros(
                        (sample.height, sample.width), dtype=np.uint8)
                    start_len = 0

                    for idx, landmark_len_item in enumerate(landmark_len):
                        cur_landmark = landmark[start_len:(
                            start_len + landmark_len_item)]
                        start_len += landmark_len_item
                        cur_keypoint_class = keypoint_cls[idx]
                        mask = self.landmark_to_mask_vec(
                            mask, cur_landmark, cur_keypoint_class)
                    landmark = mask.copy()

                    # val mode
                    if (self.is_training is False) and (
                            #   sample.h!=landmark.h or sample.w!=landmark.w
                            sample.height != landmark.shape[0] or sample.width != landmark.shape[1]):
                        # cv2.resize() params:dst.shape=(w,h)
                        landmark = cv2.resize(
                            landmark, (sample.width, sample.height))
            else:
                landmark = np.zeros(
                    (sample.height, sample.width), dtype=np.uint8)

        else:
            landmark = np.zeros((sample.height, sample.width), dtype=np.uint8)

        if self.train_transform is not None:
            sample = self.train_transform(sample)
            # sample.shape=[3,h,w]
            # landmark.shape=[h,w]
            if sample.shape[1] != landmark.shape[0] or sample.shape[2] != landmark.shape[1]:
                # cv2.resize::dsize=(w,h)
                landmark = cv2.resize(
                    landmark, (sample.shape[2], sample.shape[1]))

        # landmark = np.array(landmark, dtype=np.float32)
        # shape=[1,h,w]
        landmark = landmark[np.newaxis, :, :]

        # return sample, target, landmark
        return sample, landmark

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def landmark_to_mask_vec(
            mask: np.ndarray,
            keypoints_list: list,
            class_id: int = 1
    ) -> np.ndarray:
        mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask)
        xy = [tuple(point) for point in keypoints_list]
        assert len(xy) != 1
        draw.polygon(xy=xy, outline=0, fill=class_id)
        mask = np.array(mask, dtype=np.uint8)

        return mask
