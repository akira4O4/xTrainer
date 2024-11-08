from typing import Optional, Tuple
import torch
import torchvision.transforms as T


class BaseTransform:
    def __init__(self) -> None:
        self._mean = (0.485, 0.456, 0.406)
        self._std = (0.229, 0.224, 0.225)
        self._ops = []

    def add_to_tensor(self) -> "BaseTransform":
        self._ops.append(T.ToTensor())
        return self

    def add_normalize(
        self,
        mean: Optional[Tuple[float]] = None,
        std: Optional[Tuple[float]] = None
    ) -> "BaseTransform":
        if mean is None:
            mean = self._mean
        if std is None:
            std = self._std

        self._ops.append(T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))
        return self

    def __call__(self, data) -> torch.Tensor:
        t = T.Compose(self._ops)
        return t(data)


if __name__ == '__main__':
    bt = BaseTransform()
    bt.add_to_tensor() \
        .add_normalize()
