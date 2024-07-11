import os
from typing import Optional
import numpy as np
from dataclasses import dataclass

'''
Labelme Data

{
    "shapes": [
        {
           "label": "",
           "points": [[x,y],[x,y],...]
        },
        {...},
        ...
    ],
    "imagePath": ""
}
'''


@dataclass
class Label:
    metadata: Optional[dict] = None
    objects: Optional[list] = None
    image_path: Optional[str] = ''
    num_of_objects: Optional[int] = 0
    is_background: Optional[int] = False
    mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.metadata is not None:
            self.objects = self.metadata.get('shapes')
            self.image_path = self.metadata.get('imagePath')
            self.num_of_objects = len(self.objects)
            self.is_background = self.num_of_objects == 0
            self.metadata['imageData'] = None

    def load_metadata(self, val) -> None:
        self.metadata = val
        self.objects = self.metadata.get('shapes')
        self.image_path = self.metadata.get('imagePath')
        self.num_of_objects = len(self.objects)
        self.is_background = self.num_of_objects == 0
        self.metadata['imageData'] = None


@dataclass
class Img:
    path: Optional[str] = None
    image: Optional[np.ndarray] = None
    exists: Optional[bool] = False

    def __post_init__(self) -> None:
        if self.path is not None:
            self.exists = os.path.exists(self.path)
