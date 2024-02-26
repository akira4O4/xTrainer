from loguru import logger
from typing import Callable
from functools import wraps
import inspect
from collections import OrderedDict

__all__ = ['Register']


class Register:
    def __init__(
            self,
            name: str,
    ):
        self._name = name
        self._data = {}

    def registered(self, target):

        def add_item(key, value):
            if not callable(value):
                raise Exception(f"Error:{value} must be callable!")

            self._data[key] = value
            logger.info(f'{self._name} register: {key}')

            return value

        if callable(target):  # Function or Class
            return add_item(target.__name__, target)
        else:  # target->key
            return lambda x: add_item(target, x)

    @property
    def name(self) -> str:
        return self._name

    def get(self, key: str):
        return self._data.get(key)

    def __call__(self, target):
        return self.registered(target)

    def __getitem__(self, item):
        return self._data.get(item)

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key) -> bool:
        return key in self._data

    def __str__(self):
        return str(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()
