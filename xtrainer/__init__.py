import platform

import numpy as np
import torch
import torchvision
from xtrainer.utils.config import Config
from xtrainer.utils.task import Task

try:
    with open("VERSION", "r") as f:
        VERSION = f.read().strip()
except FileNotFoundError:
    VERSION = 'NoFoundVersion'

CONFIG = Config()
TASK = Task(CONFIG('task'))
PROJECT = Config('project')
EXPERIMENT = Config('experiment')
TOPK = Config('topk')

NoArgs = None
# DEFAULT_WORKSPACE: str = 'project'
DEFAULT_OPTIMIZER: str = 'AdamW'
COLOR_LIST: np.ndarray = np.random.uniform(0, 255, size=(80, 3)).astype(int)

OS: str = platform.system()
MACOS: bool = (OS == "Darwin")
LINUX: bool = (OS == "Linux")
WINDOWS: bool = (OS == "Windows")
CUDA: bool = torch.cuda.is_available()
TORCH_VERSION: bool = torch.__version__
TORCHVISION_VERSION: bool = torchvision.__version__
