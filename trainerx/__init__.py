import platform

import numpy as np
import torch

from trainerx.utils.config import Config

try:
    with open("VERSION", "r") as f:
        VERSION = f.read().strip()
except FileNotFoundError:
    VERSION = 'NoFoundVersion'

CONFIG = Config()

TopK: int = 2
NoArgs = None
DEFAULT_WORKSPACE: str = 'project'
DEFAULT_OPTIMIZER: str = 'AdamW'

COLOR_LIST: np.ndarray = np.random.uniform(0, 255, size=(80, 3)).astype(int)

OS: str = platform.system()
MACOS: bool = (OS == "Darwin")
LINUX: bool = (OS == "Linux")
WINDOWS: bool = (OS == "Windows")

CUDA: bool = torch.cuda.is_available()
