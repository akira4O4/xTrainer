from typing import Optional
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from imgaug import augmenters as ia
# from .iaa_augment import iaa_augment_list

__all__ = [
    # 'IAATransform',
    # 'AugKeypoints',
    'ClsTransform',
    'SegTransform'
]


class IAATransform:
    def __init__(self, iaa_aug_seq, normalize, *args, **kwargs):
        self.aug_ = iaa_aug_seq
        ops = [
            ToTensor(),
            normalize,
        ]
        ops += list(args)

        self.transforms_ = Compose(ops)

    def __call__(self, img):
        img = np.array(img)
        img = self.aug_.augment_image(img)
        img = self.transforms_(img)
        return img


class AugKeypoints(torch.nn.Module):

    def __init__(self, p, seq_det, convert_float_coord):
        super().__init__()
        self.p = p
        self.seq_det = seq_det.to_deterministic()
        self.convert_float_coord = convert_float_coord

    def forward(self, img, keypoints):
        """
        Args:
            img (PIL Image or Tensor): Image to be equalized.

        Returns:
            PIL Image or Tensor: Randomly equalized image.
        """
        self.seq_det.to_deterministic()
        if torch.rand(1).item() < self.p:
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


class BaseTransform:
    def __init__(
            self,
            mean: Optional[list] = None,
            std: Optional[list] = None,
            wh: Optional[list] = None
    ) -> None:
        self._mean = [0.485, 0.456, 0.406] if mean is None else mean
        self._std = [0.229, 0.224, 0.225] if mean is None else std
        self._Normalize = Normalize(mean=self._mean, std=self._std)
        self._Resize = Resize((wh[1], wh[0])) if wh is not None else None
        self._iaa_aug_seq = iaa_augment_list()

    @property
    def val_trans(self):
        val_transform = Compose([
            ToTensor(),
            self._Normalize
        ])
        return val_transform


class ClsTransform(BaseTransform):
    def __init__(
            self,
            mean: Optional[list] = None,
            std: Optional[list] = None,
            wh: Optional[list] = None
    ) -> None:
        super().__init__(mean, std, wh)

    @property
    def cls_image_trans(self):
        cls_train_transform = IAATransform(
            self._iaa_aug_seq,
            self._Normalize
        )
        return cls_train_transform

    @property
    def cls_target_trans(self):
        return None


class SegTransform(BaseTransform):
    def __init__(
            self,
            mean: Optional[list] = None,
            std: Optional[list] = None,
            wh: Optional[list] = None
    ) -> None:
        super().__init__(mean, std, wh)

    @property
    def seg_image_trans(self):
        ops = [
            ToTensor(),
            self._Normalize
        ]
        if self._Resize is not None:
            ops.append(self._Resize)
        seg_image_transform = Compose(ops)
        return seg_image_transform

    @property
    def seg_target_trans(self):
        seg_target_transform = AugKeypoints(
            p=1,
            seq_det=self._iaa_aug_seq,
            convert_float_coord=True
        )
        return seg_target_transform
