from typing import Optional, List, Tuple, Union
import random
import cv2
import torch
import numpy as np
from PIL import Image

from torchvision.transforms import functional as F
from imgaug import augmenters as iaa
from trainerx.utils.common import np2pil
from .functional import (
    random_hsv,
    random_flip,
    resize,
    letterbox,
    to_tensor
)


class NP2PIL:
    def __init__(self):
        ...

    def __call__(
        self,
        data: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[Image.Image, Image.Image]:
        image, mask = data
        return np2pil(image), np2pil(mask)


class RandomHSV:
    def __init__(
        self,
        h_gain: Optional[float] = 0.5,
        s_gain: Optional[float] = 0.5,
        v_gain: Optional[float] = 0.5
    ) -> None:
        assert h_gain or s_gain or v_gain
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        image, mask = data

        assert image is not None, 'image is None.'
        im_hsv = random_hsv(image, self.h_gain, self.s_gain, self.v_gain)

        return im_hsv, mask


class RandomFlip:
    def __init__(
        self,
        flip_thr: Optional[float] = 0.5,
        direction: Optional[str] = "horizontal"
    ) -> None:
        self.flip_thr = flip_thr
        self.direction = direction

    def __call__(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        image, mask = data
        random_p = random.random()
        image = random_flip(image, self.flip_thr, self.direction, random_p)
        mask = random_flip(mask, self.flip_thr, self.direction, random_p)
        return image, mask


class Resize:
    def __init__(self, wh: Tuple[int, int], only_scaledown=True):
        self.wh = wh
        self.only_scaledown = only_scaledown

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image is not None, 'image is None.'

        ih, iw = image.shape[:2]

        if (iw, ih) == self.wh:
            return image

        image = resize(image, self.wh, self.only_scaledown)
        return image


class SegResize:
    def __init__(self, wh: Tuple[int, int], only_scaledown=True):
        self.wh = wh
        self.only_scaledown = only_scaledown

    def __call__(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        image, mask = data
        assert image is not None, 'image is None.'
        assert mask is not None, 'mask is None'

        ih, iw = image.shape[:2]

        if (iw, ih) == self.wh:
            return image, mask

        image = resize(image, self.wh, self.only_scaledown)
        mask = resize(image, self.wh, self.only_scaledown)
        return image, mask


class LetterBox:
    def __init__(self, wh: Tuple[int, int], only_scaledown=True):
        self.wh = wh
        self.only_scaledown = only_scaledown

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image is not None, 'image is None.'

        ih, iw = image.shape[:2]

        if (iw, ih) == self.wh:
            return image

        image = letterbox(image, self.wh, self.only_scaledown)
        return image


class SegLetterBox:
    def __init__(self, wh: Tuple[int, int], only_scaledown=True):
        self.wh = wh
        self.only_scaledown = only_scaledown

    def __call__(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        image, mask = data
        assert image is not None, 'image is None.'
        assert mask is not None, 'mask is None'

        ih, iw = image.shape[:2]

        if (iw, ih) == self.wh:
            return image, mask

        image = letterbox(image, self.wh, self.only_scaledown)
        mask = letterbox(mask, self.wh, self.only_scaledown, (0, 0, 0))
        return image, mask


class ImgAugT:
    def __init__(self):
        self.t = iaa.Sequential(
            [
                iaa.SomeOf(
                    (0, 1),
                    [
                        iaa.Flipud(0.7),
                        iaa.Fliplr(0.7)
                    ]
                ),
                iaa.MultiplyHue((0.9, 1.1)),
                iaa.MultiplySaturation((0.9, 1.1)),
                iaa.SomeOf(
                    (0, 1),
                    [
                        iaa.Add((-10, 20)),
                        iaa.Multiply((0.8, 1.2))
                    ]
                )
            ]
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.t(image)


class ToTensor:
    def __init__(self, half: Optional[bool] = False):
        super().__init__()
        self.half = half

    def __call__(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = data
        image = to_tensor(image, self.half)
        mask = to_tensor(mask, self.half)
        return image, mask


class Normalize:
    def __init__(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ):
        self.mean = [0.485, 0.456, 0.406] if mean is None else mean
        self.std = [0.229, 0.224, 0.225] if std is None else std

    def __call__(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = data
        image = F.normalize(image, mean=self.mean, std=self.std)
        mask = F.normalize(mask, mean=self.mean, std=self.std)
        return image, mask
