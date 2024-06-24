import numpy as np
from .datafile import DataFile
import platform
import torch

with open("VERSION", "r") as f:
    VERSION = f.read().strip()

CONFIG = DataFile()

# default val ----------------------------------------------------------------------------------------------------------
NoArgs = None
Default_WorkSpace_Dir = 'workspace'
DEFAULT_WORKSPACE = 'workspace'
COLOR_LIST = np.random.uniform(0, 255, size=(80, 3))
OS = platform.system()
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
CUDA = torch.cuda.is_available()
