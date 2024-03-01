from typing import Optional
from dataclasses import dataclass, field


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

# if __name__ == '__main__':
#     a = ClassificationDataArgs()
#     print(a)
