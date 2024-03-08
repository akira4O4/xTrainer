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


def xywh2xyxy(label_data, img_w, img_h, img: np.ndarray) -> np.ndarray:
    label, x, y, w, h = label_data

    label = int(label)
    label_ind = label

    # 边界框反归一化
    x_t = x * img_w
    y_t = y * img_h
    w_t = w * img_w
    h_t = h * img_h

    # 计算坐标
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2

    p1, p2 = (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y))
    # 绘制矩形框
    cv2.rectangle(img, p1, p2, colormap[label_ind + 1], thickness=2, lineType=cv2.LINE_AA)
    label = labels[label_ind]
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=2)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # 绘制矩形框填充
        cv2.rectangle(
            img,
            p1, p2,
            colormap[label_ind + 1],
            -1,
            cv2.LINE_AA
        )
        # 绘制标签
        cv2.putText(
            img,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 2 / 3, colormap[0],
            thickness=2,
            lineType=cv2.LINE_AA
        )
    return img


def check_yolo_data(root: str):
    images = get_file_with_ext(os.path.join(root, 'images'), support_image_type)
    labels = get_file_with_ext(os.path.join(root, 'labels'), ['.txt'])

    for image, label in zip(images, labels):
        img = cv2.imread(image)
        h, w = img.shape[:2]

        with open(label, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
        # 绘制每一个目标
        for x in lb:
            # 反归一化并得到左上和右下坐标，画出矩形框
            img = xywh2xyxy(x, w, h, img)
        cv2.imshow('img', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    root = r'D:\llf\dataset\test_data\yolo'
    check_yolo_data(root)
