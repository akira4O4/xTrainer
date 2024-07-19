from typing import Optional, List, Tuple, Union
import random
import cv2
import torch
import numpy as np


def random_hsv(
    image: np.ndarray,
    h_gain: Optional[float] = 0.5,
    s_gain: Optional[float] = 0.5,
    v_gain: Optional[float] = 0.5
) -> np.ndarray:
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    dtype = image.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)

    return im_hsv


def random_flip(
    image: np.ndarray,
    flip_thr: Optional[float] = 0.5,
    direction: Optional[str] = "horizontal",
    random_p: Optional[float] = None,
) -> np.ndarray:
    assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
    assert 0 <= flip_thr <= 1.0
    assert image is not None, 'image is None.'

    random_p = random.random() if random_p is None else random_p
    if direction == "vertical" and random_p < flip_thr:
        image = np.flipud(image)

    if direction == "horizontal" and random_p < flip_thr:
        image = np.fliplr(image)

    return image


def resize(
    image: np.ndarray,
    wh: Tuple[int, int],
    only_scaledown=True
) -> np.ndarray:
    ih, iw = image.shape[:2]
    new_w, new_h = wh[0], wh[1]

    if only_scaledown:
        if (wh[1] / ih) > 1:
            new_h = ih
        if (wh[0] / iw) > 1:
            new_w = iw

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return image


def letterbox(
    image: np.ndarray,
    wh: Tuple[int, int],
    only_scaledown: Optional[bool] = True,
    pad_value: Tuple[int, int, int] = None
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
        value=(114, 114, 114) if pad_value is None else pad_value
    )
    return image


def to_tensor(
    data: np.ndarray,
    half: Optional[bool] = False
) -> torch.Tensor:
    data = np.ascontiguousarray(data)
    data = torch.from_numpy(data)  # to torch
    data = data.half() if half else data.float()
    data /= 255.0  # 0-255 to 0.0-1.0
    return data
