import os
import yaml
import json
from typing import Optional, Any, Union

from loguru import logger


class DataFile:
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._data = {}
        if path is not None and os.path.exists(path):
            self.load()

    @property
    def data(self) -> dict:
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    def clear(self) -> None:
        self._data = {}

    # a={'k':[1,2]}
    # b={'k':[3,4]}
    # a.add(b)->a={'k':[1,2,3,4]}
    def add(self, data: dict) -> None:

        for key in self._data.keys():
            if isinstance(self._data[key], list):
                self._data[key].extend(data.get(key, []))

    def load(self, path: Optional[str] = None) -> None:
        if path is not None:
            self._path = path
        base_name = os.path.basename(self._path)
        _, suffix = os.path.splitext(base_name)
        if suffix.lower() == '.yaml':
            self._load_yaml()
        elif suffix.lower() == '.json':
            self._load_json()

        else:
            logger.error('Input file is not yaml or json. ')
            exit()

    def _load_json(self) -> None:
        with open(self.path, 'r') as config_file:
            self._data = json.load(config_file)
        logger.info(f'Loading: {self.path}.')

    def _load_yaml(self) -> None:
        with open(self._path, encoding='utf-8') as f:
            self._data = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(f'Loading: {self.path}.')

    def remove(self, key: str) -> bool:
        if self._data.get(key) is None:
            logger.error(f'Key: {key}is not found.')
            return False
        else:
            logger.info(f'Del {key}:{self._data[key]}.')
            del self._data[key]
            return True

    def items(self) -> Any:
        if isinstance(self._data, dict):
            return self._data.items()
        else:
            return None

    def update(self, data: dict) -> None:
        if isinstance(self._data, dict):
            self._data.update(data)
        else:
            return None

    def save(self, output: Optional[str] = None) -> None:

        save_path = self._path if output is None else output

        basename = os.path.basename(save_path)
        name, suffix = os.path.splitext(basename)

        if suffix.lower() == '.json':
            self.save_json(save_path)
        elif suffix.lower() in ['.yaml', '.yml']:
            self.save_yaml(save_path)

    def save_json(self, output: Optional[str] = None) -> None:
        save_path = output if output is not None else self.path

        with open(save_path, 'w') as f:
            f.write(json.dumps(self.data, indent=4, ensure_ascii=False))

        logger.info(f'Save Json File in: {save_path}.')

    def save_yaml(self, output: Optional[str] = None) -> None:
        save_path = output if output is not None else self.path

        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(data=self.data, stream=f, allow_unicode=True)

        logger.info(f'Save Yaml File in: {save_path}.')

    def len(self) -> int:
        if self._data is None:
            return 0
        else:
            return len(self._data)

    def __call__(self, key: Union[str, int]) -> Any:
        if isinstance(self._data, list):
            return self._data[key]

        elif isinstance(self._data, dict):
            return self.data.get(key, None)
