from typing import Optional, List, Tuple, Union, overload
import random
import cv2
import torch
import numpy as np
from PIL import Image

from torchvision.transforms import functional as F
from imgaug import augmenters as iaa
from xtrainer.utils.common import np2pil, hwc2chw
from .functional import (
    random_hsv,
    random_flip,
    resize,
    letterbox,
)


class NP2PIL:
    @staticmethod
    def _call_2(data: Tuple[np.ndarray, np.ndarray]) -> Tuple[Image.Image, Image.Image]:
        image, mask = data
        return np2pil(image), np2pil(mask)

    @staticmethod
    def _call_1(image: np.ndarray) -> Image.Image:
        return np2pil(image)

    def __call__(self, data):
        if type(data) is np.ndarray:
            return self._call_1(data)
        elif type(data) is tuple:
            return self._call_2(data)


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
    def __init__(self, wh: Tuple[int, int], only_scaledown=False):
        self.wh = wh
        self.only_scaledown = only_scaledown

    def _call_2(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        image, mask = data
        assert image is not None, 'image is None.'
        assert mask is not None, 'mask is None'

        ih, iw = image.shape[:2]

        if (iw, ih) == self.wh:
            return image, mask

        image = resize(image, self.wh, self.only_scaledown)
        # mask = resize(image, self.wh, self.only_scaledown)
        return image, mask

    def _call_1(self, image: np.ndarray) -> np.ndarray:
        assert image is not None, 'image is None.'

        ih, iw = image.shape[:2]

        if (iw, ih) == self.wh:
            return image

        image = resize(image, self.wh, self.only_scaledown)
        return image

    def __call__(self, data):
        if type(data) is np.ndarray:
            return self._call_1(data)
        elif type(data) is tuple:
            return self._call_2(data)


class LetterBox:
    def __init__(self, wh: Tuple[int, int], only_scaledown=False):
        self.wh = wh
        self.only_scaledown = only_scaledown

    def _call_1(self, image: np.ndarray) -> np.ndarray:

        assert image is not None, 'image is None.'

        ih, iw = image.shape[:2]

        if (iw, ih) == self.wh:
            return image

        image = letterbox(image, self.wh, self.only_scaledown)
        return image

    def _call_2(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        image, mask = data
        assert image is not None, 'image is None.'
        assert mask is not None, 'mask is None'

        ih, iw = image.shape[:2]

        if (iw, ih) == self.wh:
            return image, mask

        image = letterbox(image, self.wh, self.only_scaledown)
        return image, mask

    # data: (np.ndarray,np.ndarray)
    # data: np.ndarray
    def __call__(self, data):
        if type(data) is np.ndarray:
            return self._call_1(data)
        elif type(data) is tuple:
            return self._call_2(data)


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

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        image = F.to_tensor(image)
        image = image.half() if self.half else image.float()

        mask = hwc2chw(mask)
        mask = torch.from_numpy(mask)

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
        return image, mask
