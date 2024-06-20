class DataLogger:
    def __init__(self, name):  # 4e

        self._name = name
        self._count = 0
        self._sum_of_batch_size = 0
        self._sum = 0
        self._avg = 0
        self._val = 0

        self.reset()

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    # def update(self, val, n=-1, batch_size: int = -1) -> None:
    #     self._val = val
    #     self._sum += val
    #
    #     if n != -1:
    #         self._count += 1
    #         self._avg = self._sum / self._count  # (A+B+C+...)/n
    #
    #     if batch_size != -1:
    #         self._sum_of_batch_size += batch_size
    #         self._avg = self._sum / self._sum_of_batch_size  # (A+B+C)/bs

    def update(self, val) -> None:
        self._val = val
        self._sum += val
        self._count += 1
        self._avg = self._sum / self._count

    # def update(self, val, n=1):
    #     self.val = val
    #     self.sum += val * n
    #     self.count += n
    #     self.avg = self.sum / self.count

    @property
    def sum(self) -> float:
        return float(self._sum)

    @property
    def avg(self) -> float:
        return float(self._avg)

    @property
    def name(self) -> str:
        return self._name

    @property
    def curr_val(self) -> float:
        return float(self._val)
