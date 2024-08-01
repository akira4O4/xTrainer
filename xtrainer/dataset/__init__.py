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


# Segmentation label
@dataclass
class SegLabel:
    metadata: Optional[dict] = None
    objects: Optional[list] = None
    image_path: Optional[str] = ''
    num_of_objects: Optional[int] = 0
    is_background: Optional[int] = False
    mask: Optional[np.ndarray] = None

    def _decode(self) -> None:
        if self.metadata is not None:
            self.objects = self.metadata.get('shapes')
            self.image_path = self.metadata.get('imagePath')
            self.num_of_objects = len(self.objects)
            self.is_background = self.num_of_objects == 0
            self.metadata['imageData'] = None

    def load_metadata(self, val) -> None:
        self.metadata = val
        self._decode()

    def __post_init__(self) -> None:
        self._decode()


@dataclass
class Image:
    path: Optional[str] = ''
    data: Optional[np.ndarray] = None
    exists: Optional[bool] = False

    def __post_init__(self) -> None:
        self.exists = os.path.exists(self.path)
