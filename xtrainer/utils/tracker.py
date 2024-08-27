import numpy as np


class DataTracker:
    def __init__(self, name: str) -> None:
        self.name = name
        self._metadata = []
        self._val: float = 0.0

    def reset(self) -> None:
        self._metadata = []
        self._val = 0.0

    def add(self, val) -> None:
        self._metadata.append(val)
        self._val = val

    @property
    def size(self) -> int:
        return len(self._metadata)

    @property
    def metadata(self) -> list:
        return self._metadata

    @property
    def sum(self) -> float:
        return float(np.sum(self._metadata))

    @property
    def avg(self) -> float:
        return float(np.mean(self._metadata))

    @property
    def val(self) -> float:
        return self._val

    def __len__(self) -> int:
        return self.size


class ClsTrainTracker:
    def __init__(self, name: str = 'TrainTracker', topk: int = 2):  # noqa
        self.name = name
        self.top1 = DataTracker(f'Train Top1')
        self.topk = DataTracker(f'Train Top{topk}')  # noqa
        self.loss = DataTracker('Classification Loss')

    def reset(self) -> None:
        self.top1.reset()
        self.topk.reset()
        self.loss.reset()


class ClsValTracker:
    def __init__(self, name: str = 'ValTracker', topk: int = 2):  # noqa
        self.name = name
        self.top1 = DataTracker(f'Train Top1')
        self.topk = DataTracker(f'Train Top{topk}')  # noqa
        self.loss = DataTracker('Classification Loss')

    def reset(self) -> None:
        self.top1.reset()
        self.topk.reset()
        self.loss.reset()


class SegTrainTracker:
    def __init__(self, name: str = 'Segmentation Train Tracker', topk: int = 2):  # noqa
        self.name = name
        self.miou = DataTracker(f'Train MIoU')  # noqa
        self.loss = DataTracker('Segmentation Loss')

    def reset(self) -> None:
        self.miou.reset()
        self.loss.reset()


class SegValTracker:
    def __init__(self, name: str = 'Segmentation Val Tracker', topk: int = 2):  # noqa
        self.name = name
        self.miou = DataTracker(f'Val MIoU')  # noqa
        self.loss = DataTracker('Segmentation Loss')

    def reset(self) -> None:
        self.miou.reset()
        self.loss.reset()
