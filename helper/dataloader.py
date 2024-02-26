import os
from math import ceil
from typing import List, Union, Optional

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from loguru import logger
# import tensorflow as tf
# import tensorflow_datasets as tfds
from .dataset import ClassificationDataset
from utils.util import get_images
from typing import Any, Callable, TypeVar, Generic, Sequence, List, Optional
import torch.distributed as dist

__all__ = [
    'DataLoaderWrapper'
]


# class GenClsTFRecordFile:
#     def __init__(self, root: str, output: str, num_example: int = 128, suffix: str = None):
#         self.root = root
#         self.output = output
#         self.num_example = num_example
#         self.suffix = suffix
#         self.dataset = ClassificationDataset(root)
#         self.samples = get_images(root)

#     def record(self):

#         step = ceil(len(self.dataset) / self.num_example)
#         for i in range(step):
#             output = os.path.join(self.output, str(i) + self.suffix)

#             begin_idx = i * self.num_example
#             end_idx = i * self.num_example + self.num_example

#             if end_idx > len(self.dataset):
#                 end_idx = len(self.dataset)
#             print(f'Dataset[{begin_idx}:{end_idx}]')
#             self.save_one_example(self.dataset.samples[begin_idx: end_idx], output)

#     def save_one_example(self, samples: list, output: str):
#         logger.info(f'Writing file: {output}')
#         writer = tf.io.TFRecordWriter(output)
#         for sample in samples:
#             image = cv2.imread(sample[0])
#             shape = image.shape
#             example = tf.train.Example(
#                 features=tf.train.Features(feature={
#                     "image": self.bytes_feature(image.tobytes()),
#                     "height": self.int64_feature(shape[0]),
#                     "width": self.int64_feature(shape[1]),
#                     "channels": self.int64_feature(shape[2]),
#                     "label": self.int64_feature(sample[1])
#                 })
#             )
#             writer.write(example.SerializeToString())
#         writer.close()

#     @staticmethod
#     def bytes_feature(value: bytes):
#         return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#     @staticmethod
#     def bytes_list_feature(value: List[bytes]):
#         return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

#     @staticmethod
#     def int64_feature(value: int):
#         return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#     @staticmethod
#     def int64_list_feature(value: List[int]):
#         return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

#     @staticmethod
#     def float_feature(value: float):
#         return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

#     @staticmethod
#     def float_list_feature(value: List[float]):
#         return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# # TODO:Impl this code
# class ClsTFRecordDataset(Dataset):
#     def __init__(self, filenames: Union[str, list], parse_func):
#         self.filenames = filenames
#         self.parse_func = parse_func
#         dataset = tf.data.TFRecordDataset(self.filenames)
#         dataset = dataset.map(map_func=self.parse_func)

#     @staticmethod
#     def _parse_single_example(example):
#         feature_description = {
#             'image': tf.io.FixedLenFeature([], tf.string),
#             'height': tf.io.FixedLenFeature((), tf.int64),
#             'width': tf.io.FixedLenFeature((), tf.int64),
#             'channels': tf.io.FixedLenFeature((), tf.int64),
#             'label': tf.io.FixedLenFeature([], tf.int64)
#         }
#         features = tf.io.parse_single_example(example, feature_description)
#         image = tf.decode_raw(features['image'], out_type=tf.uint8)  # decode data to tf.float32
#         height = features["height"]
#         width = features["width"]
#         channels = features["channels"]
#         label = features['label']
#         # image = tf.reshape(image, [height, width, channels])
#         return image, label

#     def __len__(self) -> int:
#         return 0

#     def __getitem__(self, idx: str):
#         ...


# # TODO:Impl this code
# class TorchTFRecordDataLoader:
#     def __init__(
#             self,
#             root: Union[str, list],
#             batch_size: int,
#             map_func=None,
#             shuffle: bool = False
#     ):
#         self.root = root
#         self.batch_size = batch_size
#         self.map_func = map_func if map_func is not None else self._parse_single_example
#         self.dataset = None
#         # self._iter = None
#         self.len = 0

#     @staticmethod
#     def _parse_single_example(example):
#         feature_description = {
#             'image': tf.io.FixedLenFeature([], tf.string),
#             'height': tf.io.FixedLenFeature((), tf.int64),
#             'width': tf.io.FixedLenFeature((), tf.int64),
#             'channels': tf.io.FixedLenFeature((), tf.int64),
#             'label': tf.io.FixedLenFeature([], tf.int64)
#         }
#         features = tf.io.parse_single_example(example, feature_description)
#         image = tf.decode_raw(features['image'], out_type=tf.uint8)  # decode data to tf.float32
#         height = features["height"]
#         width = features["width"]
#         channels = features["channels"]
#         label = features['label']
#         image = tf.reshape(image, [height, width, channels])
#         return image, label

#     def init_tfrecord_file(self):
#         self.dataset = tf.data.TFRecordDataset(self.root)
#         self.dataset = self.dataset.map(map_func=self.map_func)
#         # dataset = dataset.prefetch()
#         self.dataset = self.dataset.batch(8)
#         self.dataset = tfds.as_numpy(self.dataset)
#         dataset = iter(self.dataset)
#         self.len = 0
#         for image, label in dataset:
#             self.len += image.shape[0]

#     def __iter__(self):
#         self.init_tfrecord_file()
#         self._iter = iter(self.dataset)
#         return self.dataset

#     def __next__(self):
#         batch = next(self._iter)
#         return batch

#     def __len__(self) -> int:
#         return self.len


class BalancedBatchSampler:
    def __init__(self, labels, n_classes: int, n_samples: int):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0] for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        # 这里指的是对应lebel已经使用的数据个数
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, size=self.n_classes, replace=True)  #
            indices = []

            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                                            class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                # 当数据的使用数量大于这个lebel具有的数据长度的时候，重置已使用数据的数量
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield indices

            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size  # return num of batch


class DataLoaderWrapper:
    def __init__(
            self,
            dataset=None,
            batch_size: Optional[int] = 1,
            shuffle: Optional[bool] = False,
            num_workers: Optional[int] = 0,
            pin_memory: Optional[bool] = False,
            **kwargs
    ):
        self._dataset = dataset
        self._dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )
        self._distributed_sampler = None

    def __call__(self):
        return self._dataloader

    @property
    def step(self) -> int:
        return len(self._dataloader)

    # def get_distributed_sampler(self):
    #     self._distributed_sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
    #     return self._distributed_sampler
