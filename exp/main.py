import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
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
# im=cv2.imread()
root=r'/media/seeking/data/llf/hrsf/classification/val'
images=get_images(root)
bad_images=[]
for image in tqdm(images):
    assert os.path.exists(image)
    im = cv2.imdecode(np.fromfile(image, dtype=np.uint8), -1)

    # im=cv2.imread(image)
    if im is None:
        bad_images.append(image)
print(bad_images)
print(len(bad_images))