import random
from typing import Optional, Tuple

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Normalize
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import functional as F

from imgaug import augmenters as ia
from imgaug import augmenters as iaa

__all__ = [
    'BaseTransform',
    'ValTransform',
    'ClsImageTransform',
    'ClsTargetTransform',
    'SegImageTransform',
    'SegTargetTransform'
]


class BaseTransform:
    def __init__(self) -> None:
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]

        self._ops = [
            ToTensor(),
            Normalize(mean=self._mean, std=self._std)
        ]

    def get_compose(self):
        return Compose(self._ops)


class ValTransform(BaseTransform):
    def __init__(self):
        super(ValTransform, self).__init__()

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        assert type(img) == np.ndarray
        t = self.get_compose()
        img = t(img)
        return img


class ClsImageTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self.iaa_aug_seq = iaa.Sequential(
            [
                iaa.SomeOf((0, 1), [iaa.Flipud(0.7), iaa.Fliplr(0.7)]),
                iaa.MultiplyHue((0.9, 1.1)),
                iaa.MultiplySaturation((0.9, 1.1)),
                iaa.SomeOf((0, 1), [iaa.Add((-10, 20)), iaa.Multiply((0.8, 1.2))]),
            ]
        )

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        assert type(img) == np.ndarray

        img = self.iaa_aug_seq.augment_image(img)
        t = self.get_compose()
        img = t(img)
        return img


class ClsTargetTransform:
    def __init__(self) -> None:
        ...

    def __call__(self, data):
        return data


class SegImageTransform(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        assert type(img) == np.ndarray
        t = self.get_compose()
        img = t(img)
        return img


class SegTargetTransform(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, keypoint):
        return keypoint


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

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            mask = np.flipud(mask)
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        return img, mask


class LetterBox:
    def __init__(
        self,
        new_hw: Tuple[int],
        auto=False,
        scale_fill=False,
        only_scaledown=True,
        center=True,
        stride=32
    ):
        self.new_hw = new_hw
        self.new_h = new_hw[0]
        self.new_w = new_hw[1]

        self.auto = auto
        self.scale_fill = scale_fill
        self.only_scaledown = only_scaledown
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, image: np.ndarray = None):

        ih, iw = image.shape[:2]

        # Scale ratio (new / old)
        r = min(self.new_h / ih, self.new_w / iw)
        # only scale down, do not scale up (for better val mAP)
        if self.only_scaledown:
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(iw * r)), int(round(ih * r))
        dw, dh = iw - new_unpad[0], ih - new_unpad[1]  # wh padding

        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding

        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (self.new_w, self.new_h)
            ratio = self.new_w / iw, self.new_h / ih  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        return image


def letterbox(image: np.ndarray, wh: Tuple[int, int], pad_color: Optional[tuple] = None) -> tuple:
    src_h, src_w = image.shape[:2]
    dst_w, dst_h = wh
    scale = min(dst_h / src_h, dst_w / src_w)
    pad_h, pad_w = int(round(src_h * scale)), int(round(src_w * scale))

    if image.shape[0:2] != (pad_w, pad_h):
        out = cv2.resize(image, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
    else:
        out = image

    top = int((dst_h - pad_h) / 2)
    down = int((dst_h - pad_h + 1) / 2)
    left = int((dst_w - pad_w) / 2)
    right = int((dst_w - pad_w + 1) / 2)

    # add border
    out = cv2.copyMakeBorder(out, top, down, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    x_offset, y_offset = max(left, right) / dst_w, max(top, down) / dst_h
    return out, x_offset, y_offset
