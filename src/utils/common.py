import os
import time
import json
import yaml
import shutil
from typing import Optional, List


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
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if clean:
            shutil.rmtree(path)
            os.makedirs(path)
