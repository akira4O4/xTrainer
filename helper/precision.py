class DataPrecision:
    def __init__(self) -> None:
        self._Max: int = 12
        self._Highest: int = 8
        self._High: int = 6
        self._Medium: int = 4
        self._Low: int = 2

    @property
    def Max(self) -> int:
        return self._Max

    @property
    def High(self) -> int:
        return self._High

    @property
    def Hightest(self) -> int:
        return self._Highest

    @property
    def Medium(self) -> int:
        return self._Medium

    @property
    def Low(self) -> int:
        return self._Low


data_precision = DataPrecision()