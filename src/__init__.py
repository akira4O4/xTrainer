import numpy as np
from .config import Config
import platform
import torch

with open("VERSION", "r") as f:
    VERSION = f.read().strip()

CONFIG = Config()

# default val ----------------------------------------------------------------------------------------------------------
NoArgs = None
Default_WorkSpace_Dir = 'workspace'
DEFAULT_WORKSPACE = 'project'
DEFAULT_OPTIMIZER = 'AdamW'
COLOR_LIST = np.random.uniform(0, 255, size=(80, 3))
OS = platform.system()
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
CUDA = torch.cuda.is_available()
