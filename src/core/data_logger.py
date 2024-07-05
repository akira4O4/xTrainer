class DataLogger:
    def __init__(self, name: str) -> None:
        self.name = name
        self._metadata = []
        self._size = 0
        self._sum = 0.0
        self._avg = 0.0
        self._val = 0.0

    def reset(self) -> None:
        self._metadata = []
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._size = 0

    def add(self, val) -> None:
        self._metadata.append(val)
        self._val = val
        self._sum += val
        self._size += 1
        self._avg = self._sum / self._size

    @property
    def metadata(self) -> list:
        return self._metadata

    @property
    def sum(self) -> float:
        return float(self._sum)

    @property
    def size(self) -> int:
        return self._size

    @property
    def avg(self) -> float:
        return float(self._avg)

    @property
    def val(self) -> float:
        return float(self._val)


class TrainLogger:
    def __init__(self, name: str = 'TrainLogger', topk: int = 2):  # noqa
        self.name = name
        self.top1 = DataLogger(f'Train Top1')
        self.topk = DataLogger(f'Train Top{topk}')  # noqa
        self.miou = DataLogger(f'Train MIoU')  # noqa


class ValLogger:
    def __init__(self, name: str = 'ValLogger', topk: int = 2):  # noqa
        self.name = name
        self.top1 = DataLogger(f'Val Top1')
        self.topk = DataLogger(f'Val Top{topk}')  # noqa
        self.miou = DataLogger(f'Val MIoU')  # noqa


class LossLogger:
    def __init__(self, name: str = 'Loss'):
        self.name = name
        self.cross_entropy_loss = DataLogger('CrossEntropy Loss')
        self.period_loss = DataLogger('Period Loss')
        self.dice_loss = DataLogger('Dice Loss')
