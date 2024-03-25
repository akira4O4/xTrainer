import os
import cv2
from utils.util import get_images
from tqdm import tqdm

if __name__ == '__main__':
    root = 'D:\llf\dataset\jushi\cam_0_1_bmp\cam1'
    output = 'D:\llf\dataset\jushi\cam_0_1_jpg\cam1'
    if not os.path.exists(output):
        os.makedirs(output)
    for image in tqdm(get_images(root, ['.bmp'])):
        basename = os.path.basename(image)
        name, ext = os.path.splitext(basename)
        im = cv2.imread(image)
        cv2.imwrite(os.path.join(output, name + '.jpg'), im)
