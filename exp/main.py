from typing import Callable
import torch


def func1():
    print('1')


def func2():
    print('2')


def func3():
    print('3')


funcs = {
    'func1': func1,
    'func2': func2,
    'func3': func3,
}


def a(a=100, b=200, c=300):
    print(a, b, c)


# for k, v in funcs.items():
#     print(k)
#     v()
print(a())
d = {'c': 1, 'a': 2, 'b': 3}
# print(a(**d))
