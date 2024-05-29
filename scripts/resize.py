import cv2
import os
from utils.util import get_images
from tqdm import tqdm
from PIL import Image
import numpy as np

if __name__ == '__main__':
    root = r'C:\Users\Lee Linfeng\Desktop\E_good\928x928'
    output = r'C:\Users\Lee Linfeng\Desktop\E_good\output'
    wh = (576, 576)
    output = output + f'-w{wh[0]}h{wh[1]}'

    if not os.path.exists(output):
        os.makedirs(output)

    images = get_images(root)
    for image in tqdm(images):
        img = Image.open(image)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        img = cv2.resize(img, wh)

        basename = os.path.basename(image)
        save_path = os.path.join(output, basename)
        # print(save_path)
        cv2.imwrite(save_path, img)
