from typing import Union, Optional, Callable
from dataclasses import dataclass, field

import torch


# @dataclass
# class SegmentationDataSetArgs:
#     root: str
#     loader: str = 'pil'
#     add_background: bool = True
#     transform: Optional[Callable] = None  # to samples
#     target_transform: Optional[Callable] = None  # to target
#     is_training: Optional[bool] = False
#     expanding_rate: Optional[int] = 0
#     img_type: Optional[str] = 'RGB'
#
#
# @dataclass
# class SegmentationTrainAndVal:
#     train: SegmentationDataSetArgs
#     val: SegmentationDataSetArgs


@dataclass
class ProjectArgs:
    task: str
    work_dir: str


@dataclass
class TrainArgs:
    seed: Optional[int] = 0
    deterministic: Optional[bool] = False
    epoch: Optional[int] = 0
    max_epoch: int = 0
    print_freq: int = 10
    workflow: list = field(default_factory=list)


@dataclass
class ModelArgs:
    model_name: str
    num_classes: int = 0
    mask_classes: int = 0
    pretrained: Optional[bool] = False
    model_path: Optional[str] = None
    gpu: int = 0,
    strict: Optional[bool] = True
    map_location: Optional[str] = 'cpu'
    input_channels: Optional[int] = 3


# @dataclass
# class DataLoaderArgs:
#     dataset: Callable
#     batch_size: Optional[int] = 1
#     shuffle: bool = False
#     num_workers: int = 0
#     pin_memory: bool = False
#
#
# @dataclass
# class ClassificationDataSetArgs:
#     root: str
#     wh: Optional[Union[list, tuple]] = None
#     loader: str = 'pil'
#     transform: Optional[Callable] = None
#     target_transform: Optional[Callable] = None
#     expanding_rate: Optional[int] = 0
#     letterbox: Optional[bool] = False
#     img_type: Optional[str] = 'RGB'
#
#
# @dataclass
# class ClassificationDataSetTrainAndVal:
#     train: ClassificationDataSetArgs
#     val: ClassificationDataSetArgs
#
#
# @dataclass
# class DataloaderTrainAndVal:
#     train: DataLoaderArgs
#     val: DataLoaderArgs
#
#
# @dataclass
# class ClassificationDataArgs:
#     dataset: ClassificationDataSetTrainAndVal
#     dataloader: DataloaderTrainAndVal


def auto_loading_config(config):
    project_args = ProjectArgs(**config['project_config'])
    train_args = TrainArgs(**config['train_config'])
    model_args = ModelArgs(**config['model_config'])
    print(project_args)
    print(train_args)
    print(model_args)


# if __name__ == '__main__':
#     a = ClassificationDataArgs()
#     print(a)
