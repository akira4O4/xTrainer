import numpy as np

from typing import List, Union, Optional
from dataclasses import dataclass


class Labels:
    def __init__(self, labels: List[str]):
        self._data: List[str] = labels
        self._size: int = len(labels)

    @property
    def labels(self) -> List[str]:
        return self._data

    @property
    def size(self) -> int:
        return self._size

    def idx2str(self, idx: int) -> str:
        if idx < 0 or idx >= self._size:
            raise IndexError("索引超出范围")
        return self._data[idx]

    def str2idx(self, name: str) -> int:
        if name not in self._data:
            raise ValueError(f"'{name}' 不在标签列表中")
        return self._data.index(name)

    def __getitem__(self, item: Union[int, str]) -> Union[int, str]:
        if isinstance(item, int):
            return self.idx2str(item)
        elif isinstance(item, str):
            return self.str2idx(item)
        else:
            raise TypeError("输入必须是整数或字符串")


# only support labelme format
class MaskLabel:
    def __init__(self, metadata: Optional[dict] = None):

        self.metadata: Optional[dict] = None
        self.objects: Optional[list] = None
        self.image_path: Optional[str] = ''
        self.num_objects: Optional[int] = 0
        self.is_background: Optional[int] = False
        self.ih: Optional[int] = 0
        self.iw: Optional[int] = 0
        self.mask: Optional[np.ndarray] = None

        if metadata is not None:
            self.metadata = metadata
            self._decode()

    def set_metadata(self, val) -> None:
        self.metadata = val
        self._decode()

    def _decode(self) -> None:
        if self.metadata is not None:
            self.objects = self.metadata.get('shapes')
            self.image_path = self.metadata.get('imagePath')
            self.iw = self.metadata.get('imageWidth')
            self.ih = self.metadata.get('imageHeight')

            self.num_objects = len(self.objects)
            self.is_background = self.num_objects == 0
            self.metadata['imageData'] = None
