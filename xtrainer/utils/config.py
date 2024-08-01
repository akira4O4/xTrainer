import os
from typing import Optional, Any, Union
from xtrainer.utils.common import load_yaml, load_json


class Config:
    def __init__(self, path: Optional[str] = None) -> None:

        self._path: str = ''
        self._metadata = None

        if path is not None:
            if os.path.exists(path):
                self._path = path
                self.load()

    def __call__(self, key: Union[str, int]) -> Any:
        if isinstance(self._metadata, list):
            return self._metadata[key]

        elif isinstance(self._metadata, dict):
            return self._metadata.get(key, None)

    @property
    def metadata(self):
        return self._metadata

    def set_metadata(self, val) -> None:
        self._metadata = val

    def set_path(self, path: str) -> None:
        self._path = path

    def load(self) -> None:
        basename = os.path.basename(self._path)
        name, subfix = os.path.splitext(basename)
        if subfix == '.yaml' or subfix == '.yml':
            self._metadata = load_yaml(self._path)
        elif subfix == '.json':
            self._metadata = load_json(self._path)
