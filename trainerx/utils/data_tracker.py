import numpy as np
from trainerx import TopK


class DataTracker:
    def __init__(self, name: str) -> None:
        self.name = name
        self._metadata = []
        self._val = 0.0

    def reset(self) -> None:
        self._metadata = []
        self._val = 0

    def add(self, val) -> None:
        self._metadata.append(val)
        self._val = val

    def __len__(self) -> int:
        return self.size

    @property
    def size(self) -> int:
        return len(self._metadata)

    @property
    def metadata(self) -> list:
        return self._metadata

    @property
    def sum(self):  # return type: (int ,float)
        return np.sum(self._metadata).tolist()

    @property
    def avg(self):  # return type: float
        return np.mean(self._metadata).tolist()

    @property
    def val(self):
        return self._val


class TrainTracker:
    def __init__(self, name: str = 'TrainLogger', topk: int = TopK):  # noqa
        self.name = name
        self.top1 = DataTracker(f'Train Top1')
        self.topk = DataTracker(f'Train Top{topk}')  # noqa
        self.miou = DataTracker(f'Train MIoU')  # noqa


class ValTracker:
    def __init__(self, name: str = 'ValLogger', topk: int = TopK):  # noqa
        self.name = name
        self.top1 = DataTracker(f'Val Top1')
        self.topk = DataTracker(f'Val Top{topk}')  # noqa
        self.miou = DataTracker(f'Val MIoU')  # noqa


# class LossTracker:
#     def __init__(self, name: str = 'Loss'):
#         self.name = name
#         self.cross_entropy_loss = DataTracker('CrossEntropy Loss')
#         self.period_loss = DataTracker('Period Loss')
#         self.dice_loss = DataTracker('Dice Loss')
