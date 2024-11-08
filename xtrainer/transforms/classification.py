from base import BaseTransform
from typing import Optional, Tuple
import numpy as np
import torchvision.transforms as T
from PIL import Image


class ClassificationTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def add_resize(self, hw: Tuple[int, int], *args, **kwargs) -> "ClassificationTransform":
        self._ops.append(T.Resize(hw, *args, **kwargs))
        return self

    def add_letterbox(self) -> "ClassificationTransform":
        return self

    def add_random_resized_crop(self, hw: Tuple[int, int], *args, **kwargs) -> "ClassificationTransform":
        self._ops.append(T.RandomResizedCrop(hw, *args, **kwargs))
        return self

    def add_random_horizontal_flip(self, p: Optional[float] = 0.5) -> "ClassificationTransform":
        if p > 0:
            self._ops.append(T.RandomHorizontalFlip(p))
        return self

    def add_random_vertical_flip(self, p: Optional[float] = 0.5) -> "ClassificationTransform":
        if p > 0:
            self._ops.append(T.RandomVerticalFlip(p))
        return self

    def add_color_jitter(self, *args, **kwargs) -> "ClassificationTransform":
        self._ops.append(T.ColorJitter(*args, **kwargs))
        return self

    def add_rand_augment(self, *args, **kwargs) -> "ClassificationTransform":
        self._ops.append(T.RandAugment(*args, **kwargs))
        return self

    def add_augmix(self, *args, **kwargs) -> "ClassificationTransform":
        self._ops.append(T.AugMix(*args, **kwargs))
        return self

    def add_auto_augment(self, *args, **kwargs) -> "ClassificationTransform":
        self._ops.append(T.AutoAugment(*args, **kwargs))
        return self

    def add_random_erasing(self, p: Optional[float] = 0.5, *args, **kwargs) -> "ClassificationTransform":
        if p > 0:
            self._ops.append(T.RandomErasing(p=p, inplace=True, *args, **kwargs))
        return self

    def add_center_crop(self, hw: Tuple[int, int]) -> "ClassificationTransform":
        self._ops.append(T.CenterCrop(hw))
        return self


if __name__ == '__main__':
    interpolation = T.InterpolationMode.BILINEAR

    ct = ClassificationTransform()
    ct.add_random_resized_crop((255, 255)) \
        .add_random_horizontal_flip(0.5) \
        .add_random_vertical_flip(0.5) \
        .add_rand_augment(interpolation=interpolation) \
        .add_augmix(interpolation=interpolation) \
        .add_auto_augment(interpolation=interpolation) \
        .add_color_jitter(0.4, 0.4, 0.7, 0.015) \
        .add_to_tensor() \
        .add_normalize() \
        .add_random_erasing(0.4)

    width, height = 416, 416
    random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(random_data, 'RGB')
    out = ct(image)
