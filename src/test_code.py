from typing import Callable
import torch


def func1():
    print(1)


def func2():
    print(2)


aa = {
    '1': func1(),
    '2': func2()
}
aa.get('1')
