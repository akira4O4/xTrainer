from typing import Tuple

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from trainerx.augment.transforms import (
    RandomFlip,
    RandomHSV,
    SegLetterBox,
    NP2PIL,
    SegNP2PIL,
    LetterBox,
    ToTensor,
    Normalize
)


# Base Transform -------------------------------------------------------------------------------------------------------

class BaseT:
    def __init__(self) -> None:
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._ops = []
        self._default_ops = [
            T.ToTensor(),
            T.Normalize(mean=self._mean, std=self._std)
        ]

    def gen_compose(self) -> T.Compose:
        return T.Compose(self._ops)

    def __call__(self, image) -> torch.Tensor:
        t = self.gen_compose()
        return t(image)


# Classification Transform ---------------------------------------------------------------------------------------------
class ClsImageT(BaseT):
    def __init__(self, wh: Tuple[int, int]):
        super().__init__()
        assert wh is not None, 'imgsz is not None.'
        hw = (wh[1], wh[0])
        self._ops = [
            LetterBox(wh),
            NP2PIL(),
            T.RandomResizedCrop(hw),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomErasing(inplace=True),
            T.RandAugment(interpolation=Image.BILINEAR),  # noqa
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015),
        ]
        self._ops += self._default_ops

        self.t = self.gen_compose()

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        img = self.t(image)
        return img


class ClsTargetT(BaseT):
    def __init__(self) -> None:
        super().__init__()
        ...

    def __call__(self, data: int) -> int:
        return data


class ClsValT(BaseT):
    def __init__(self, wh: Tuple[int, int]) -> None:
        super().__init__()
        assert wh is not None, 'imgsz is not None.'
        self._ops = [
            LetterBox(wh),
            NP2PIL()
        ]
        self._ops += self._default_ops
        self.t = self.gen_compose()

    def __call__(self, image) -> torch.Tensor:
        image = self.t(image)
        return image


# Segmentation Transform -----------------------------------------------------------------------------------------------
class SegImageT(BaseT):
    def __init__(self, wh: Tuple[int, int]) -> None:
        super().__init__()
        assert wh is not None, 'imgsz is not None.'
        self._ops = [
            SegLetterBox(wh),
            RandomHSV(),
            RandomFlip(direction="vertical"),
            RandomFlip(direction="horizontal"),
            SegNP2PIL(),
            ToTensor(),
            Normalize()
        ]
        self.t = self.gen_compose()

    def __call__(
        self,
        data: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.t(data)
        return image, mask


class SegValT(BaseT):
    def __init__(self, wh: Tuple[int, int]) -> None:
        super().__init__()
        assert wh is not None, 'imgsz is not None.'
        self._ops = [
            SegLetterBox(wh),
            NP2PIL(),
            ToTensor(),
            Normalize()
        ]
        self.t = self.gen_compose()

    def __call__(
        self,
        data: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = data
        image, mask = self.t((image, mask))
        return image, mask
