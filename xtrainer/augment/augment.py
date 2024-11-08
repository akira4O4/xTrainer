from typing import Optional, List, Tuple
import random
import torch
import numpy as np
from torchvision.transforms import functional as F
from imgaug import augmenters as iaa
from functional import (
    random_hsv,
    random_flip,
    resize,
    letterbox,
    hwc2chw
)


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
            flip_p: Optional[float] = 0.5,
            direction: Optional[str] = "horizontal"
    ) -> None:
        self.flip_p = flip_p
        self.direction = direction

    def __call__(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        image, mask = data
        random_p = random.random()
        image = random_flip(image, self.flip_p, self.direction, random_p)
        mask = random_flip(mask, self.flip_p, self.direction, random_p)
        return image, mask


class Resize:
    def __init__(self, wh: Tuple[int, int], only_scaledown=False):
        self.wh = wh
        self.only_scaledown = only_scaledown

    def _impl(self, image: np.ndarray) -> np.ndarray:
        ih, iw = image.shape[:2]

        if (iw, ih) == self.wh:
            return image

        image = resize(image, self.wh, self.only_scaledown)
        return image

    def __call__(self, data: Tuple[np.ndarray, np.ndarray]):
        image, mask = data
        assert image is not None, 'image is None.'
        assert mask is not None, 'mask is None'

        new_image = self._impl(image)
        return new_image, mask


class LetterBox:
    def __init__(self, wh: Tuple[int, int], only_scaledown=False):
        self.wh = wh
        self.only_scaledown = only_scaledown

    def _impl(self, image: np.ndarray) -> np.ndarray:

        ih, iw = image.shape[:2]

        if (iw, ih) == self.wh:
            return image

        image = letterbox(image, self.wh, self.only_scaledown)
        return image

    def __call__(self, data):
        if isinstance(data, np.ndarray):
            assert data is not None, 'image is None.'
            return self._impl(data)

        elif isinstance(data, tuple):
            image, mask = data
            assert image is not None, 'image is None.'
            assert mask is not None, 'mask is None'
            new_image = self._impl(image)
            return new_image, mask


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
        self.mean = mean
        self.std = std

    def __call__(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = data
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask
