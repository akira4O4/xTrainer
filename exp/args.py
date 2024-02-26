from typing import Union, Optional, Callable
from dataclasses import dataclass


@dataclass
class ClassificationDataSetArgs:
    root: str
    wh: Optional[Union[list, tuple]] = None
    loader: str = 'pil'
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    expanding_rate: Optional[int] = 0
    letterbox: Optional[bool] = False
    img_type: Optional[str] = 'RGB'


@dataclass
class SegmentationDataSetArgs:
    root: str
    loader: str = 'pil'
    add_background: bool = True
    transform: Optional[Callable] = None  # to samples
    target_transform: Optional[Callable] = None  # to target
    is_training: Optional[bool] = False
    expanding_rate: Optional[int] = 0
    img_type: Optional[str] = 'RGB'


@dataclass
class ClassificationTrainAndVal:
    train: ClassificationDataSetArgs
    val: ClassificationDataSetArgs


@dataclass
class SegmentationTrainAndVal:
    train: SegmentationDataSetArgs
    val: SegmentationDataSetArgs


@dataclass
class DataSetArgs:
    classification: ClassificationTrainAndVal
    segmentation: SegmentationTrainAndVal


@dataclass
class DataLoaderArgs:
    dataset: Callable
    batch_size: Optional[int] = 1
    shuffle: bool = False
    sampler: Optional[Callable] = None,
    batch_sampler: Optional[Callable] = None
    num_workers: int = 0
    collate_fn: Optional[Callable] = None
    pin_memory: bool = False
    kwargs: dict = None


@dataclass
class ModelArgs:
    model_name: str
    num_classes: Optional[int] = 0
    mask_classes: Optional[int] = 0
    pretrained: Optional[bool] = False
    model_path: Optional[str] = None
    gpu: Optional[int] = -1,  # default (-1)==cpu
    strict: Optional[bool] = True
    map_location: Optional[str] = 'cpu'
    input_channels: Optional[str] = 3


if __name__ == '__main__':
    dataset = DataSetArgs
    dataset.classification.train = None
    dataset.classification.val = None
