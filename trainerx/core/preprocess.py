import random
from typing import Optional, Tuple

import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from imgaug import augmenters as iaa

__all__ = [
    'BaseT',
    'ValidateT',
    'ClsImageT',
    'ClsTargetT',
    'SegImageT',
    'SegTargetT',
    'letterbox',
    'LetterBox'
]


# Augmentation Transform -----------------------------------------------------------------------------------------------

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

    def __call__(self, img: np.ndarray) -> np.ndarray:
        r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)

        return im_hsv


class RandomFlip:
    def __init__(
        self,
        p: Optional[float] = 0.5,
        direction: Optional[str] = "horizontal",
    ) -> None:
        assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0
        self.p = p
        self.direction = direction

    def __call__(
        self,
        image: np.ndarray,
        target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.direction == "vertical" and random.random() < self.p:
            image = np.flipud(image)
            target = np.flipud(image) if target is not None else target

        if self.direction == "horizontal" and random.random() < self.p:
            image = np.fliplr(image)
            target = np.fliplr(image) if target is not None else target

        return image, target


def letterbox(
    image: np.ndarray,
    wh: Tuple[int, int],
    only_scaledown: Optional[bool] = True
) -> np.ndarray:
    assert isinstance(image, np.ndarray) is True, 'input image.type must be np.ndarray.'
    ih, iw = image.shape[:2]

    new_w, new_h = wh[0], wh[1]
    # Min scale ratio (new / old)
    r = min(new_h / ih, new_w / iw)

    # only scale down, do not scale up (for better val mAP)
    if only_scaledown:
        r = min(r, 1.0)

    # Compute padding
    # ratio = r, r  # width, height ratios
    pad_w, pad_h = int(round(iw * r)), int(round(ih * r))
    dw, dh = iw - pad_w, ih - pad_h  # wh padding

    dw /= 2
    dh /= 2

    if [ih, iw] != [pad_h, pad_w]:  # resize
        image = cv2.resize(image, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    image = cv2.copyMakeBorder(
        image,
        top, bottom,
        left, right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )
    return image


class LetterBox:
    def __init__(
        self,
        wh: Tuple[int, int],
        only_scaledown=True,
    ):
        self.wh = wh
        self.only_scaledown = only_scaledown

    def __call__(self, image: np.ndarray = None) -> np.ndarray:
        return letterbox(image, self.wh, self.only_scaledown)


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


# Base Transform -------------------------------------------------------------------------------------------------------

class BaseT:
    def __init__(self) -> None:
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._ops = []
        self._default_ops = [
            T.ToTensor(),
            T.Normalize(mean=self._mean, std=self._std),
            T.RandomErasing(p=0.4, inplace=True)
        ]

    def get_compose(self) -> T.Compose:
        return T.Compose(self._ops)

    def __call__(self, image) -> torch.Tensor:
        t = self.get_compose()
        return t(image)


class ValidateT(BaseT):
    ...


# Test Transform -------------------------------------------------------------------------------------------------------
class ImageT(BaseT):
    def __init__(self):
        super().__init__()

    def __call__(self, image) -> torch.Tensor:
        return image


class TargetT(BaseT):
    def __init__(self) -> None:
        super().__init__()
        ...

    def __call__(self, target) -> torch.Tensor:
        return target


# Classification Transform ---------------------------------------------------------------------------------------------
class ClsImageT(BaseT):
    def __init__(self, imgsz: int):
        self.imgsz = imgsz
        super().__init__()

        self._ops = [
            T.RandomResizedCrop(self.imgsz),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomErasing(inplace=True),
            T.RandAugment(interpolation=Image.BILINEAR),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015),
        ]
        self._ops += self._default_ops

        self.t = self.get_compose()

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        img = self.t(image)
        return img


class ClsTargetT(BaseT):
    def __init__(self) -> None:
        super().__init__()
        ...

    def __call__(self, data):
        return data


# Segmentation Transform -----------------------------------------------------------------------------------------------
# TODO: re design
class SegImageT(BaseT):
    def __init__(self) -> None:
        super().__init__()
        self._ops = [RandomHSV()]
        self._ops.extend(self._default_ops)
        self.t = self.get_compose()

    def __call__(self, image) -> torch.Tensor:
        image = self.t(image)
        return image


# TODO: re design
class SegTargetT(BaseT):
    def __init__(self) -> None:
        super().__init__()
        self._ops.extend(self._default_ops)
        self.t = self.get_compose()

    def __call__(self, image) -> torch.Tensor:
        image = self.t(image)
        return image


if __name__ == '__main__':
    r = np.random.uniform(-1, 1, 3) * [0.5, 0.5, 0.5] + 1
    print(r)
