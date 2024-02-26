from PIL import Image
from tqdm import tqdm
import os
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

if __name__ == '__main__':
    # im=Image.open('/media/seeking/data/llf/gzsapa_xiansao/classification/train/5_others/daoguanyichang/0/Board0_Time_20230301-17_21_47.972target1.png')
    # exit(
    images = get_images(r'/media/seeking/data/llf/gzsapa_xiansao')
    # images = get_images(r'/home/seeking/0')
    print(len(images))
    for image in tqdm(images):
        try:
            img = Image.open(image)
            img=img.convert('RGB')
        except:
            print(image)
            