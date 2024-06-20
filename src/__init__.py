import numpy as np
from .datafile import DataFile

NoArgs = None

with open("VERSION", "r") as f:
    VERSION = f.read().strip()

CONFIG = DataFile()
Default_WorkSpace_Dir = 'workspace'
COLOR_LIST = np.random.uniform(0, 255, size=(80, 3))
