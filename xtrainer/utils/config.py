import os
from typing import Optional, Any, Union, Dict
from xtrainer.utils.common import load_yaml, load_json


class Config:
    def __init__(self, path: Optional[str] = None) -> None:
        self._path: str = path
        self._metadata: Dict[str, Any] = {}

    def __getitem__(self, key: str):
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

    def set_path(self, path: str) -> None:
        if os.path.exists(path):
            self._path = path
        else:
            raise FileNotFoundError(f"The path {path} does not exist.")

    def update(self, kv: Dict[str, Any]) -> None:
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
