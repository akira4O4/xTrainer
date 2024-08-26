import os
from typing import Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class Image:
    path: Optional[str] = ''
    data: Optional[np.ndarray] = None
    exists: Optional[bool] = False

    def __post_init__(self) -> None:
        self.exists = os.path.exists(self.path)
