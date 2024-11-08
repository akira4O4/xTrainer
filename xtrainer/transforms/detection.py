from base import BaseTransform
from typing import Optional, Tuple
import numpy as np
from augment.augment import RandomHSV, RandomFlip, ToTensor, Normalize, Resize, LetterBox


class DetectionTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def add_resize(self, wh: Tuple[int, int], only_scaledown=False) -> "DetectionTransform":
        self._ops.append(Resize(wh, only_scaledown))
        return self

    def add_letterbox(self, wh: Tuple[int, int], only_scaledown=False) -> "DetectionTransform":
        self._ops.append(LetterBox(wh, only_scaledown))
        return self

    def add_random_hsv(
        self,
        h_gain: Optional[float] = 0.5,
        s_gain: Optional[float] = 0.5,
        v_gain: Optional[float] = 0.5
    ) -> "DetectionTransform":
        self._ops.append(RandomHSV(h_gain, s_gain, v_gain))
        return self

    def add_random_horizontal_flip(self, p: Optional[float] = 0.5) -> "DetectionTransform":
        if p > 0:
            self._ops.append(RandomFlip(p, "horizontal"))
        return self

    def add_random_vertical_flip(self, p: Optional[float] = 0.5) -> "DetectionTransform":
        if p > 0:
            self._ops.append(RandomFlip(p, "vertical"))
        return self

    def add_to_tensor(self, half: Optional[bool] = False) -> "DetectionTransform":
        self._ops.append(ToTensor(half))
        return self

    def add_normalize(
        self,
        mean: Optional[Tuple[float]] = None,
        std: Optional[Tuple[float]] = None
    ) -> "DetectionTransform":

        if mean is None:
            mean = self._mean
        if std is None:
            std = self._std

        self._ops.append(Normalize(mean, std))
        return self


if __name__ == '__main__':
    dt = DetectionTransform()
    dt.add_resize((224, 224)) \
        .add_random_hsv() \
        .add_random_vertical_flip(0.5) \
        .add_random_horizontal_flip(0.5) \
        .add_to_tensor() \
        .add_normalize()

    width, height = 416, 416
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    label = image
    data = (image, label)
    output = dt(data)
