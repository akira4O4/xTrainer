import os
import math
import time
import json
import yaml
import shutil
from typing import Optional, List, Tuple, Union
import numpy as np
from PIL import Image
from xtrainer.utils.emoji import Emoji


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def round4(data: float) -> float:
    assert isinstance(data, float)
    return round(float(data), 4)


def round8(data: float) -> float:
    assert isinstance(data, float)
    return round(float(data), 8)


def load_json(path: str):
    with open(path, 'r') as config_file:
        data = json.load(config_file)
    return data


def load_yaml(path: str):
    with open(path, encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def save_json(data, save: str) -> None:
    with open(save, 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))


def save_yaml(data, save: str) -> None:
    with open(save, 'w', encoding='utf-8') as f:
        yaml.dump(data=data, stream=f, allow_unicode=True)


def get_images(path: str, ext: Optional[List[str]] = None) -> List[str]:
    ext = ['.png', '.jpg'] if ext is None else ext
    data = []

    for root, dirs, files in os.walk(path):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext in ext:
                image = os.path.join(root, file)
                data.append(image)
    return data


def get_json_file(path: str) -> List[str]:
    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            name, suffix = os.path.splitext(file)
            if suffix.lower() == '.json':
                image = os.path.join(root, file)
                data.append(image)
    return data


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print(f'🕐 {func.__name__} Spend Time: {format(time_spend, ".3f")}s')
        return result

    return func_wrapper


def get_time(fmt: str = '%Y%m%d_%H%M%S') -> str:
    time_str = time.strftime(fmt, time.localtime())
    return str(time_str)


def error_exit() -> None:
    exit(1)


def check_dir(path: str, clean: bool = False) -> None:
    if os.path.exists(path) is False:
        os.makedirs(path)
    else:
        if clean:
            shutil.rmtree(path)
            os.makedirs(path)


def get_image_shape(image: Union[np.ndarray, Image.Image]) -> Tuple[int, int]:
    img_w, img_h = -1, -1
    if isinstance(image, Image.Image):
        img_w, img_h = image.size
    elif isinstance(image, np.ndarray):
        img_h, img_w, c = image.shape
    return img_w, img_h


def check_size(image: np.ndarray, wh: Tuple[int, int]) -> bool:
    img_w, img_h = get_image_shape(image)
    if img_w != wh[0] or img_h != wh[1]:
        return False
    else:
        return True


def pil2np(img: Image.Image) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img

    return np.asarray(img)


def np2pil(img: np.ndarray) -> Image.Image:
    if isinstance(img, Image.Image):
        return img

    return Image.fromarray(img)


def align_size(size1: int, size2: int) -> Tuple[int, int]:
    assert size1 != 0
    assert size2 != 0

    if size1 == size2:
        return 1, 1

    data = [size1, size2]
    exp_r1 = 1
    exp_r2 = 1

    max_idx = np.argmax([size1, size2])
    min_idx = np.argmin([size1, size2])
    r = math.ceil(data[max_idx] / data[min_idx])

    if size1 > size2:
        exp_r1 = 1
        exp_r2 *= r
    else:
        exp_r1 *= r
        exp_r2 = 1

    return exp_r1, exp_r2


def chw2hwc(image: np.ndarray) -> np.ndarray:
    assert len(image.shape) == 3, f'input shape must be (c,h,w)'
    return image.transpose(1, 2, 0)


def hwc2chw(image: np.ndarray) -> np.ndarray:
    assert len(image.shape) == 3, f'input shape must be (h,w,c)'
    return image.transpose(2, 0, 1)


def hw_to_1hw(image: np.ndarray) -> np.ndarray:
    assert len(image.shape) == 2, f'input shape must be (h,w)'
    return image[None]


def hw_to_hw1(image: np.ndarray) -> np.ndarray:
    assert len(image.shape) == 2, f'input shape must be (h,w)'
    return image[:, :, None]


def safe_round(data: np.ndarray, n: int = 0) -> np.ndarray:
    """
    Safely round an array to `n` decimal places, using half-up rounding.

    Parameters:
        arr (np.ndarray): Input array to be rounded.
        n (int): Number of decimal places to round to. Default is 0.

    Returns:
        np.ndarray: Rounded array.
    """

    # Scale array for rounding
    scaling_factor = 10 ** (n + 1)
    scaled_arr = data * scaling_factor

    # Perform rounding
    rounded_arr = np.floor(scaled_arr + 0.5)  # Add 0.5 for correct rounding
    result = rounded_arr / scaling_factor

    # Apply original sign
    return result


def print_of_cls(
    mode: str,
    task: str,
    epoch: int,
    epochs: int,
    cls_loss: float = None,
    lr: float = None,
    top1: float = None,
    topk: float = None,
    width: int = 12
):
    epoch_progress = f'{epoch}/{epochs}'

    if mode == 'train':
        print(
            f'{Colors.GREEN}'
            f'{"Mode":>{width}}'
            f'{"Task":>{width}}'
            f'{"Epoch":>{width}}'
            f'{"cls_loss":>{width}}'
            f'{"LR":>{width}}'
            f'{"Top1":>{width}}'
            f'{"TopK":>{width}}'
            f'{Colors.ENDC}'
        )
        print(
            f'{Colors.BLUE}'
            f'{mode.upper():>{width}}'
            f'{Colors.ENDC}'
            f'{task:>{width}}'
            f'{epoch_progress:>{width}}'
            f'{cls_loss:>{width}}'
            f'{lr:>{width}}'
            f'{top1:>{width}}'
            f'{topk:>{width}}'
        )

    elif mode == 'val':
        print(
            f'{Colors.BLUE}'
            f'{mode.upper():>{width}}'
            f'{Colors.ENDC}'
            f'{"-":>{width}}'
            f'{"-":>{width}}'
            f'{"-":>{width}}'
            f'{"-":>{width}}'
            f'{top1:>{width}}'
            f'{topk:>{width}}\n'
        )


def print_of_seg(
    mode: str,
    task: str,
    epoch: int,
    epochs: int,
    seg_loss: float = None,
    lr: float = None,
    miou: float = None,
    width: int = 12
):
    epoch_progress = f'{epoch}/{epochs}'

    if mode == 'train':
        print(
            f'{Colors.GREEN}'
            f'{"Mode":>{width}}'
            f'{"Task":>{width}}'
            f'{"Epoch":>{width}}'
            f'{"seg_loss":>{width}}'
            f'{"LR":>{width}}'
            f'{"MIoU":>{width}}'
            f'{Colors.ENDC}'
        )
        print(
            f'{Colors.BLUE}'
            f'{mode.upper():>{width}}'
            f'{Colors.ENDC}'
            f'{task:>{width}}'
            f'{epoch_progress:>{width}}'
            f'{seg_loss:>{width}}'
            f'{lr:>{width}}'
            f'{miou:>{width}}'
        )
    elif mode == 'val':
        print(
            f'{Colors.BLUE}'
            f'{mode.upper():>{width}}'
            f'{Colors.GREEN}'
            f'{"-":>{width}}'
            f'{"-":>{width}}'
            f'{"-":>{width}}'
            f'{"-":>{width}}'
            f'{miou:>{width}}\n'
        )


def print_of_mt(
    mode: str,
    task: str,
    epoch: int,
    epochs: int,
    cls_loss: float = None,
    seg_loss: float = None,
    lr: float = None,
    top1: float = None,
    topk: float = None,
    miou: float = None,
    width: int = 12
):
    epoch_progress = f'{epoch}/{epochs}'

    if mode == 'train':
        print(
            f'{Colors.GREEN}'
            f'{"Mode":>{width}}'
            f'{"Task":>{width}}'
            f'{"Epoch":>{width}}'
            f'{"cls_loss":>{width}}'
            f'{"seg_loss":>{width}}'
            f'{"LR":>{width}}'
            f'{"Top1":>{width}}'
            f'{"TopK":>{width}}'
            f'{"MIoU":>{width}}'
            f'{Colors.ENDC}'
        )
        print(
            f'{Colors.BLUE}'
            f'{mode.upper():>{width}}'
            f'{Colors.ENDC}'
            f'{task:>{width}}'
            f'{epoch_progress:>{width}}'
            f'{cls_loss:>{width}}'
            f'{seg_loss:>{width}}'
            f'{lr:>{width}}'
            f'{top1:>{width}}'
            f'{topk:>{width}}'
            f'{miou:>{width}}'
        )
    elif mode == 'val':
        print(
            f'{Colors.BLUE}'
            f'{mode.upper():>{width}}'
            f'{Colors.ENDC}'
            f'{"-":>{width}}'
            f'{"-":>{width}}'
            f'{"-":>{width}}'
            f'{"-":>{width}}'
            f'{"-":>{width}}'
            f'{top1:>{width}}'
            f'{topk:>{width}}'
            f'{miou:>{width}}\n'
        )


if __name__ == '__main__':
    arr = np.array([-0.1234, -0.5678, -0.5555, -0.1, 0.1234, 0.5678, 0.5555, 0.1])
    rounded_arr = safe_round(arr, n=2)
    print("Rounded array:", rounded_arr)
