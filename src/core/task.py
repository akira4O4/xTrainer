from enum import Enum, unique


@unique
class Task(Enum):
    CLS = 'classification'
    SEG = 'segmentation'
    MultiTask = 'multitask'


def task_convert(task: str) -> Task:
    if task == 'multitask':
        return Task.MultiTask
    elif task == 'classification':
        return Task.CLS
    elif task == 'segmentation':
        return Task.SEG
