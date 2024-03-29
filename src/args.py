from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ProjectArgs:
    task: str
    work_dir: str
    mlflow_uri: Optional[str] = ''
    mlflow_port: Optional[int] = 5000
    mlflow_experiment_name: Optional[str] = "demo"


@dataclass
class TrainArgs:
    topk: Optional[int] = 2
    seed: Optional[int] = 0
    deterministic: Optional[bool] = False
    epoch: Optional[int] = 0
    max_epoch: int = 0
    print_freq: int = 10
    workflow: list = field(default_factory=list)
    amp: bool = True
    accumulation_steps: Optional[int] = 0


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


@dataclass
class TestArgs:
    test_dir: str
    weight: str
    experiment_time: str
    need_resize: bool
    good_idx: int
    sum_method: bool
    need_segment: bool
    cls_threshold: list
    seg_threshold: list
