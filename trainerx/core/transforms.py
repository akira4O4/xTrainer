from typing import Optional

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Normalize
from torchvision.transforms import Compose, ToTensor
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


class LetterBox:

    def __init__(
        self,
        new_shape: tuple = (640, 640),  # HW
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32
    ):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, image: np.ndarray) -> np.ndarray:
        shape = image.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding

        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (self.new_shape[1], self.new_shape[0])
            ratio = self.new_shape[1] / shape[1], self.new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            image,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )  # add border
        return img


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


class AugKeypoints(torch.nn.Module):  # noqa

    def __init__(self, p, seq_det, convert_float_coord):
        super().__init__()
        self.p = p
        self.seq_det = seq_det.to_deterministic()
        self.convert_float_coord = convert_float_coord

    def forward(self, img, keypoints):  # noqa
        """
        Args:
            img (PIL Image or Tensor): Image to be equalized.

        Returns:
            PIL Image or Tensor: Randomly equalized image.
        """
        self.seq_det.to_deterministic()
        if torch.rand(1).item() < self.p:

            if not isinstance(img, np.ndarray):
                img = np.asarray(img)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            new_keypoints = []
            cur_keypoints = []

            for cur_point in keypoints:
                cur_keypoints.append(ia.Keypoint(x=cur_point[0], y=cur_point[1]))

            images_aug = self.seq_det.augment_images([img])[0]
            key_points_on_Image = ia.KeypointsOnImage(cur_keypoints, shape=img.shape)
            keypoints_aug = self.seq_det.augment_keypoints([key_points_on_Image])[0]

            for i in range(len(key_points_on_Image.keypoints)):
                point_aug = keypoints_aug.keypoints[i]

                new_keypoints.append((np.array([point_aug.x, point_aug.y])).tolist())

            images_aug = cv2.cvtColor(images_aug, cv2.COLOR_BGR2RGB)
            images_aug = Image.fromarray(images_aug)
            return images_aug, new_keypoints

        return img, keypoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


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
