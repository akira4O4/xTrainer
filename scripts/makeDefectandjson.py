import cv2
import numpy as np
import random
import itertools
import os
import PIL
from PIL import Image

import io, json
import base64
from base64 import b64encode


# 类型转换
def img_tobyte(img_pil):
    ENCODING = 'utf-8'
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format='PNG')
    binary_str2 = img_byte.getvalue()
    imageData = base64.b64encode(binary_str2)
    base64_string = imageData.decode(ENCODING)
    return base64_string


def mask2json(json_path: str, imgend, label, img, mask):
    JSON_INFO = {
        "version": "4.5.9",
        "flags": {},
        "shapes": [],
        "imagePath": {},
        "imageData": {},
        "imageHeight": 0.0,
        "imageWidth": 0.0
    }

    SEG_INFO = {
        "label": "1",
        'points': [],
        "group_id": None,
        "shape_type": "polygon",
        "flags": {}
    }

    height, width = mask.shape[:2]
    JSON_INFO["imageHeight"] = height
    JSON_INFO["imageWidth"] = width

    JSON_INFO["imageData"] = img_tobyte(Image.fromarray(img))

    # saveflag = False
    # for index in range(mask.shape[2]):
    #     _, binary = cv2.threshold(mask[..., index], 128, 255, cv2.THRESH_BINARY)
    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    if binary.sum() == 0:
        return False
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lanmark_lists = []
    continue_count = 0
    for i, cnt in enumerate(contours):
        # cnt = cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True)*0.005, True)
        lanmark_len = len(cnt)
        if lanmark_len <= 2:
            print("lanmark点数小于等于2个！")
            continue_count += 1
            continue

        lanmark_lists.append([[]])
        for j in range(lanmark_len):
            lanmark_lists[i - continue_count][0].append(list(cnt[j][0].astype(np.float64)))

    if not lanmark_lists:
        return False

    saveflag = True
    # JSON_INFO["shapes"] = [] # 清空
    for i, lanmark_list in enumerate(lanmark_lists):
        SEG_INFO["label"] = label
        SEG_INFO["points"] = lanmark_list[0]
        JSON_INFO["shapes"].append(SEG_INFO.copy())

    if not saveflag:
        return False

    JSON_INFO["imagePath"] = os.path.basename(os.path.splitext(json_path)[0]) + imgend
    with open(json_path, "w") as f:
        json.dump(JSON_INFO, f)
    return True


def seek_image(root: str):
    path_list = []
    for i, j, k in os.walk(root):
        for l in k:
            if l.split('.')[-1] in {'jpg', 'jpeg', 'bmp', 'png'}:
                path = os.path.join(i, l)
                path_list.append(path)
    return path_list


def generate_defect(image,
                    location=None,
                    color_center=(0, 0, 0),
                    min_area=25,
                    max_area=75,
                    max_iterations=1,
                    mode='block',
                    ):
    """
    1.位置随机
    2.颜色随机
    3.大小随机
    4.数量随机
    
    image:待生成缺陷的bgr图像;
    location:当为None时，生成的缺陷为图像的任意位置；需要指定区域时，location传入[x,y,w,h];
    color_center:缺陷部位的bgr颜色;
    min_area:缺陷的最小面积;
    max_area:缺陷的最大面积;
    max_iterations:缺陷的最大数量;
    mode:缺陷类型,'block'为块状缺陷,'line'为线条状缺陷;
    """

    def generate_pix(pix, threshold):
        upper_limit = min(255, pix + threshold)
        lower_limit = max(0, pix - threshold)
        return random.randint(lower_limit, upper_limit)

    def calculate_distance(color1, color2):
        return np.sqrt(np.sum(np.square(np.asarray(color1, dtype=np.float32) - \
                                        np.asarray(color2, dtype=np.float32))))

    def generate_color(color, threshold=10):
        distance = float('inf')
        while distance > threshold:
            B, G, R = list(map(lambda x: generate_pix(x, threshold), color))
            distance = calculate_distance(color, (B, G, R))
        return B, G, R

    def generate_position(width, height):
        reserved_width = 5
        if width <= 10 or height <= 10:
            return width // 2, height // 2
        x = random.randint(reserved_width, width - reserved_width - 1)
        y = random.randint(reserved_width, height - reserved_width - 1)
        return x, y

    def link_points(image, point_collections):
        for p1, p2 in itertools.permutations(point_collections, 2):
            cv2.line(image, p1, p2, 127, 1)

    def generate_block_mask(image, location, min_area, max_area, init_mask=None):
        image_h, image_w, _ = image.shape
        min_side_length = min(image_h, image_w)
        bg_width = 50
        again = True
        while again:
            bg = np.zeros((bg_width, bg_width), np.uint8)
            point_cnt = random.randint(6, 20)
            point_collections = [generate_position(bg_width, bg_width) for _ in range(point_cnt)]
            link_points(bg, point_collections)
            _, bg, _, _ = cv2.floodFill(bg, None, (0, 0), 255)
            _, bg = cv2.threshold(bg, 200, 255, 1)
            area = np.count_nonzero(bg)
            random_area = random.randint(min_area, max_area)
            resize_ratio = (random_area / area) ** 0.5
            if bg_width * resize_ratio > min_side_length:
                print('block is bigger than original image,regenerate defect block...')
                continue
            bg = cv2.resize(bg, None, fx=resize_ratio, fy=resize_ratio)
            bg_h, bg_w = bg.shape

            if init_mask is None:
                mask = np.zeros((image_h, image_w), np.uint8)
            else:
                mask = init_mask.copy()
            if location is None:
                random_w, random_h = generate_position(image_w - bg_w, image_h - bg_h)
                break
            else:
                x, y, w, h = location
                random_w, random_h = generate_position(w, h)
                random_w += x
                random_h += y
                if 0 <= random_w + bg_w <= image_w and 0 <= random_h + bg_h <= image_h:
                    break
        mask[random_h:random_h + bg_h, random_w:random_w + bg_w] = bg
        return mask

    def generate_line_mask(image, location, min_area, max_area, init_mask=None):
        image_h, image_w, _ = image.shape
        area = 0
        while area < min_area or area > max_area:
            if init_mask is None:
                mask = np.zeros((image_h, image_w), np.uint8)
            else:
                mask = init_mask.copy()
            point_cnt = 4
            if location is None:
                collections = [generate_position(image_w, image_h) \
                               for _ in range(point_cnt)]
                min_y, max_y = 0, image_h
            else:
                x, y, w, h = location
                collections = [generate_position(w, h) \
                               for _ in range(point_cnt)]
                collections = list(map(lambda i: (i[0] + x, i[1] + y), collections))
                min_y, max_y = y, y + h
            collections.sort()
            point_x = list(map(lambda i: i[0], collections))
            point_y = list(map(lambda i: i[1], collections))
            poly = np.poly1d(np.polyfit(point_x, point_y, 2))
            points = []
            for x_value in range(point_x[0], point_x[-1]):
                y_value = np.int(poly(x_value))
                if min_y < y_value < max_y:
                    points.append((x_value, y_value))
                else:
                    break
            points = np.asarray(points).reshape((1, -1, 1, 2)).astype(np.int32)
            # cv2.polylines(mask, points, False, 255,thickness=0.1)
            cv2.polylines(mask, points, False, 255)
            area = cv2.countNonZero(mask)
        return mask

    def add_defect_block_to_image(mask, image, color_center):
        merged = np.where(mask[:, :, None], generate_color(color_center), image).astype(np.uint8)
        contours, _ = cv2.findContours(mask, 1, 2)
        image_h, image_w = mask.shape
        expand = 7
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < image_w and h < image_h:
                filtering_area = merged[max(0, y - expand):min(y + h + expand, image_h),
                                 max(0, x - expand):min(x + w + expand, image_w),
                                 :]
                merged[max(0, y - expand):min(y + h + expand, image_h),
                max(0, x - expand):min(x + w + expand, image_w), :] = \
                    cv2.GaussianBlur(filtering_area, (5, 5), 0)
        return merged

    def add_defect_line_to_image(mask, image, color_center):
        merged = np.where(mask[:, :, None], generate_color(color_center), image).astype(np.uint8)
        contours, _ = cv2.findContours(mask, 1, 2)
        image_h, image_w = mask.shape
        expand = 7
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < image_w and h < image_h:
                filtering_area = merged[max(0, y - expand):min(y + h + expand, image_h),
                                 max(0, x - expand):min(x + w + expand, image_w),
                                 :]
                merged[max(0, y - expand):min(y + h + expand, image_h),
                max(0, x - expand):min(x + w + expand, image_w), :] = \
                    cv2.GaussianBlur(filtering_area, (5, 5), 0)
        return merged

    image_copy = image.copy()
    iterations = random.randint(1, max_iterations)
    mask = None
    if mode.lower() == 'block':
        for _ in range(iterations):
            mask = generate_block_mask(image, location, min_area, max_area, mask)
        image_copy = add_defect_block_to_image(mask, image_copy, color_center)
    else:
        for _ in range(iterations):
            mask = generate_line_mask(image, location, min_area, max_area, mask)
        image_copy = add_defect_line_to_image(mask, image_copy, color_center)
    return image_copy, mask


def show(*images):
    for index, image in enumerate(images, 1):
        cv2.imshow('image' + str(index), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def imread(imagePath):
    image = PIL.Image.open(imagePath)
    return np.asarray(image)


def imwrite(image):
    #    caseImage = Image.fromarray(case.astype(np.uint8))
    #    caseImage.save('E:/camera/1/%d.png'%frame_index)
    image = PIL.Image.fromarray(image.astype(np.uint8))
    image.save(dstPath)
    return image


if __name__ == '__main__':
    # black_color = (20, 20, 20)
    black_color = (200, 200, 200)
    imageRoot = r'C:\Users\Administrator\Desktop\tmp\input-576\T1'
    pathList = seek_image(imageRoot)
    dstRoot = r'C:\Users\Administrator\Desktop\tmp\output-all-1-w'

    if not os.path.exists(dstRoot):
        os.makedirs(dstRoot)

    times = 5  # 同一张图片生成多少张不同的图像
    label = "1_heidian"  # label: None 没有json, 需要json，写标签例如"PM"
    boundary = 16  # 值越小边界越松
    # x1, y1 = 150, 100
    # x2, y2 = 230, 390

    x1, y1 = 0,154
    x2, y2 = 420,576
    w = x2 - x1
    h = y2 - y1
    location = [x1, y1, w, h]
    # location = None
    for imagePath in pathList:
        image = imread(imagePath)

        # display
        # draw = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # cv2.imshow('image', draw)
        # cv2.waitKey(0)
        # continue

        for i in range(times):
            image_gray, mask = generate_defect(image,
                                               location=location,
                                               # location=None,
                                               color_center=black_color,
                                               min_area=9,
                                               max_area=25,
                                               max_iterations=1,
                                               mode='block',  # 线条line，块状block
                                               )
            basename = os.path.basename(imagePath)

            os.makedirs(dstRoot, exist_ok=True)
            dstPath = os.path.join(dstRoot,
                                   os.path.splitext(basename)[0] + "_" + str(i) + os.path.splitext(basename)[1])
            imwrite(image_gray)
            if not label is None:
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                mask[mask >= boundary] = 255
                mask[mask < boundary] = 0

                keypath, imgend = os.path.splitext(dstPath)
                json_path = keypath + ".json"
                mask2json(json_path, imgend, label, image_gray, mask)
