import numpy as np
from trainerx import TopK


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


class TrainTracker:
    def __init__(self, name: str = 'TrainTracker', topk: int = TopK):  # noqa
        self.name = name
        self.top1 = DataTracker(f'Train Top1')
        self.topk = DataTracker(f'Train Top{topk}')  # noqa
        self.miou = DataTracker(f'Train MIoU')  # noqa


class ValTracker:
    def __init__(self, name: str = 'ValTracker', topk: int = TopK):  # noqa
        self.name = name
        self.top1 = DataTracker(f'Val Top1')
        self.topk = DataTracker(f'Val Top{topk}')  # noqa
        self.miou = DataTracker(f'Val MIoU')  # noqa


class LossTracker:
    def __init__(self, name: str = 'LossTracker'):
        self.name = name
        self.classification = DataTracker('Classification')
        self.segmentation = DataTracker('Segmentation')
