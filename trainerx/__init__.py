import platform

import numpy as np
import torch

from trainerx.utils.config import Config

try:
    with open("VERSION", "r") as f:
        VERSION = f.read().strip()
except:
    VERSION = 'NoFoundVersion'

CONFIG = Config()

NoArgs = None
DEFAULT_WORKSPACE = 'project'
DEFAULT_OPTIMIZER = 'AdamW'
COLOR_LIST = np.random.uniform(0, 255, size=(80, 3))
OS = platform.system()
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
CUDA = torch.cuda.is_available()
