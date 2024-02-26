import cv2
import numpy as np
import sys
from utils.util import get_file_with_ext, load_json


def calc_area(points: list) -> int:
    point_num = len(points)
    if point_num < 3: return 0
    s = points[0][1] * (points[point_num - 1][0] - points[1][0])
    for i in range(1, point_num):
        s += points[i][1] * (points[i - 1][0] - points[(i + 1) % point_num][0])
    return abs(s / 2.0)


if __name__ == '__main__':

    path = r'D:\llf\dataset\gz_sapa_0\gz_sapa_jiayaosai_front\segmentation\train\yiwu\23030329'
    files = get_file_with_ext(path, ext=['.json'])
    label = ''
    min_area = sys.maxsize
    max_area = -1
    sum_area = 0
    cnt = 0
    for file in files:
        label_data = load_json(file)
        shapes = label_data.get('shapes')

        if shapes is not None:
            for shape in shapes:
                if label == shape.get('label'):
                    cnt += 1
                    points = shape.get('points')
                    # curr_area = calc_area(points)
                    contour = np.array(points, dtype=np.int32)
                    curr_area = cv2.contourArea(contour=contour)

                    if curr_area > max_area: max_area = curr_area

                    if curr_area < min_area: min_area = curr_area

                    sum_area += curr_area

    print(f'max area:{max_area},min area:{min_area},avg area:{sum_area / cnt}')
