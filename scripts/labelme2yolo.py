import os
import shutil
import json
import cv2
import numpy as np
from tqdm import tqdm

colormap = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 255),
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255)
]  # BGR

labels = ['A', 'B', 'C']
support_image_type = ['.jpg', '.png', '.jpeg']


def get_file_with_ext(path: str, ext: list = None):
    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if ext is not None:
                file_name, file_ext = os.path.splitext(file)
                if file_ext in ext:
                    image = os.path.join(root, file)
                    data.append(image)
            else:
                image = os.path.join(root, file)
                data.append(image)
    return data


def load_json(path: str):
    data = None
    with open(path, 'r') as config_file:
        data = json.load(config_file)  # 配置字典
    return data


def convert(image_size: list, box: list) -> list:
    # box=[x0,x1,y0,y1]
    dw = 1. / image_size[0]
    dh = 1. / image_size[1]

    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]

    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return [x, y, w, h]


def json2txt(json_data, labels: list) -> list:
    """
    convert [xmin, xmax, ymin, ymax] to [x_centre, y_centre, w, h]
    """
    image_size = [json_data['imageWidth'], json_data['imageHeight']]
    shapes: list = json_data['shapes']
    shape: dict
    objs = []
    for shape in shapes:
        label = shape['label']
        points: list = shape['points']

        box = [0, 0, 0, 0]
        box[0], box[1] = points[0][0], points[1][0],
        box[2], box[3] = points[0][1], points[1][1],

        obj = convert(image_size, box)
        cls_id = labels.index(label)
        obj.insert(0, cls_id)
        objs.append(obj)
    return objs


if __name__ == '__main__':

    root = r'D:\llf\dataset\test_data'
    labelme_data_path = os.path.join(root, 'labelme')

    assert os.path.exists(root) is True, 'root is not found.'
    assert os.path.exists(labelme_data_path) is True, 'labelme path is not found.'

    yolo_images_dir = os.path.join(root, 'yolo', 'images')
    yolo_labels_dir = os.path.join(root, 'yolo', 'labels')
    if os.path.exists(yolo_images_dir) is False:
        os.makedirs(yolo_images_dir)
    if os.path.exists(yolo_labels_dir) is False:
        os.makedirs(yolo_labels_dir)

    images = get_file_with_ext(labelme_data_path, support_image_type)
    json_files = get_file_with_ext(labelme_data_path, ['.json'])

    for image in tqdm(images):
        shutil.copy(image, yolo_images_dir)

    for json_file in tqdm(json_files):
        json_data = load_json(json_file)
        objs = json2txt(json_data, labels)

        basename = os.path.basename(json_file)
        yolo_label_path = os.path.join(yolo_labels_dir, basename.replace('json', 'txt'))

        for obj in objs:
            with open(yolo_label_path, 'a') as f:
                cls_id, x, y, w, h = obj
                yolo_data = f'{str(cls_id)} {x} {y} {w} {h}\n'
                f.write(yolo_data)
