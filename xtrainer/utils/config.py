import os
from typing import Optional, Any, Union, Dict
from xtrainer.utils.common import load_yaml, load_json


class Config:
    def __init__(self, path: Optional[str] = None) -> None:
        self._path: str = ''
        self._metadata: Dict[str, Any] = {}

        if path and os.path.exists(path):
            self._path = path
            self.load()

    def __call__(self, key: str) -> Any:
        keys = key.split('.')  # data(a.b.c)=>data[a][b][c]
        value = self._metadata
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return None

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def set_metadata(self, val: Dict[str, Any]) -> None:
        if isinstance(val, dict):
            self._metadata = val
        else:
            raise ValueError("Metadata must be a dictionary.")

    def set_path(self, path: str) -> None:
        if os.path.exists(path):
            self._path = path
        else:
            raise FileNotFoundError(f"The path {path} does not exist.")

    def add_kw(self, kv: Dict[str, Any]) -> None:
        if isinstance(kv, dict):
            self._metadata.update(kv)
        else:
            raise ValueError("The argument must be a dictionary.")

    def load(self) -> None:
        _, subfix = os.path.splitext(self._path)
        if subfix in ['.yaml', '.yml']:
            self._metadata = load_yaml(self._path)
        else:
            raise ValueError("Unsupported file format. Only YAML is supported.")
