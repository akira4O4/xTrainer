import os
import time
import json
import yaml


def load_json(self) -> None:
    with open(self.path, 'r') as config_file:
        self._data = json.load(config_file)


def load_yaml(self) -> None:
    with open(self._path, encoding='utf-8') as f:
        self._data = yaml.load(f, Loader=yaml.FullLoader)


def save_json(data, save: str) -> None:
    with open(save, 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))


def save_yaml(data, save: str) -> None:
    with open(save, 'w', encoding='utf-8') as f:
        yaml.dump(data=data, stream=f, allow_unicode=True)


def get_images(path: str, ext=None) -> list:
    if ext is None:
        ext = ['.png', '.jpg']
    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext in ext:
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
