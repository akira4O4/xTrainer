from typing import Optional, List, Tuple, Union
import random
import cv2
import torch
import numpy as np
from PIL import Image

from torchvision import transforms as T
from torchvision.transforms import functional as F


class Resize:
    def __init__(self, size: List[int]):  # hw
        self.size = size

    def __call__(
        self,
        image: Union[Image.Image, torch.Tensor],
        target=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = F.resize(image, self.size)
        if target is not None:
            target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = [0.485, 0.456, 0.406] if mean is None else mean
        self.std = [0.229, 0.224, 0.225] if std is None else std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return {'image': image, 'mask': mask}


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
        return image, target


if __name__ == '__main__':
    transform = Compose([
        Resize([640, 640]),
        ToTensor(),
        Normalize()
    ])
