from typing import Tuple, Optional

import torch
import numpy as np
import torchvision.transforms as T

from xtrainer.augment.transforms import (
    RandomFlip,
    RandomHSV,
    NP2PIL,
    LetterBox,
    ToTensor,
    Normalize
)


# Base Transform -------------------------------------------------------------------------------------------------------

class BaseT:
    def __init__(self) -> None:
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.ops = []
        self.default_ops = [
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ]

    def gen_compose(self) -> T.Compose:
        return T.Compose(self.ops)

    def __call__(self, image) -> torch.Tensor:
        t = self.gen_compose()
        return t(image)


# Classification Transform ---------------------------------------------------------------------------------------------
class ClsImageT(BaseT):
    def __init__(self, wh: Tuple[int, int], only_scaledown: bool = False):
        super().__init__()
        assert wh is not None, 'imgsz is not None.'
        hw = (wh[1], wh[0])
        self.ops = [
            LetterBox(wh, only_scaledown),
            NP2PIL(),
            T.RandomResizedCrop(hw),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),  # noqa
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015),
        ]
        self.ops += self.default_ops
        self.ops += [T.RandomErasing(inplace=True)]

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
    def __init__(self, wh: Tuple[int, int], only_scaledown: bool = False) -> None:
        super().__init__()
        assert wh is not None, 'imgsz is not None.'
        self.ops = [
            LetterBox(wh, only_scaledown),
            NP2PIL()
        ]
        self.ops += self.default_ops
        self.t = self.gen_compose()

    def __call__(self, image) -> torch.Tensor:
        image = self.t(image)
        return image


# Segmentation Transform -----------------------------------------------------------------------------------------------
class SegImageT(BaseT):
    def __init__(
        self,
        wh: Tuple[int, int],
        half: Optional[bool] = False,
        only_scaledown: bool = False
    ) -> None:
        super().__init__()
        assert wh is not None, 'imgsz is not None.'
        self.ops = [
            LetterBox(wh, only_scaledown),
            RandomHSV(),
            RandomFlip(direction="vertical"),
            RandomFlip(direction="horizontal"),
            ToTensor(half),
            Normalize()
        ]
        self.t = self.gen_compose()

    def __call__(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.t(data)
        return image, mask


class SegValT(BaseT):
    def __init__(self, wh: Tuple[int, int], only_scaledown: bool = False) -> None:
        super().__init__()
        assert wh is not None, 'imgsz is not None.'
        self.ops = [
            LetterBox(wh, only_scaledown),
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
