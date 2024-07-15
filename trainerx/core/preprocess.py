import random
from typing import Optional, Tuple

import cv2
import torch
import numpy as np
import torchvision.transforms as T
from imgaug import augmenters as iaa

__all__ = [
    'BaseTransform',
    'ValidateT',
    'ClsImageT',
    'ClsTargetT',
    'SegImageT',
    'SegTargetT'
]


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


class LetterBox:
    def __init__(
        self,
        wh: Tuple[int],
        only_scaledown=True,
    ):
        self.new_w = wh[0]
        self.new_h = wh[1]
        self.only_scaledown = only_scaledown

    def __call__(self, image: np.ndarray = None) -> np.ndarray:

        ih, iw = image.shape[:2]

        # Min scale ratio (new / old)
        r = min(self.new_h / ih, self.new_w / iw)

        # only scale down, do not scale up (for better val mAP)
        if self.only_scaledown:
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


class BaseTransform:
    def __init__(self) -> None:
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._ops = []
        self._default_ops = [
            T.ToTensor(),
            T.Normalize(mean=self._mean, std=self._std)
        ]

    def get_compose(self) -> T.Compose:
        return T.Compose(self._ops)

    def __call__(self, image) -> torch.Tensor:
        t = self.get_compose()
        return t(image)


class ValidateT(BaseTransform):
    ...


class ClsImageT(BaseTransform):
    def __init__(self):
        super().__init__()
        self.iaa_t = ImgAugT()

        self._ops = [
            RandomHSV(),
            RandomFlip()
        ]
        self._ops.extend(self._default_ops)

        self.t = self.get_compose()

    def __call__(self, image) -> torch.Tensor:
        img = self.iaa_t(image)
        img = self.t(img)
        return img


class ClsTargetT(BaseTransform):
    def __init__(self) -> None:
        super().__init__()
        ...

    def __call__(self, data):
        return data


# TODO: re design
class SegImageT(BaseTransform):
    def __init__(self) -> None:
        super().__init__()
        self._ops = [RandomHSV()]
        self._ops.extend(self._default_ops)

    def __call__(self, image) -> torch.Tensor:
        t = self.get_compose()
        image = t(image)
        return image


# TODO: re design
class SegTargetT(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data) -> torch.Tensor:
        return data


if __name__ == '__main__':
    r = np.random.uniform(-1, 1, 3) * [0.5, 0.5, 0.5] + 1
    print(r)
