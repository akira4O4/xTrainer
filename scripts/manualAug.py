import imgaug.augmenters as iaa
import cv2, numpy as np
import os
import shutil
from tqdm import tqdm

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.


flag = 1  # 0good 1bad
if flag == 0:
    # good
    seq = iaa.Sequential([
        iaa.SomeOf((2, 2), [iaa.AddToHueAndSaturation((-3, 3)),  # 将图像转换至HSV空间，改变H和S值。
                            iaa.Grayscale((0.0, 0.1), 'BGR'),  # 第一个参数：转换后的图像的可见度；第二个参数：原图像的读取方式。
                            iaa.ContrastNormalization((0.95, 1.05)),  # 表示以0.8到1.2之间的随机因子来改变图像的对比度，参数值可修改。
                            # iaa.Add((1, 5), per_channel=0.5), #为图像中所有像素增加值，此例中的值为-20到20之间。
                            iaa.Add((1, 10), per_channel=0.5),  # 为图像中所有像素增加值，此例中的值为-20到20之间。
                            iaa.Multiply((0.9, 1.1), per_channel=0.5),
                            # 乘法增强器，将图像中所有像素乘以特定值，使图像更亮或者更暗，第二个参数表示作用的通道数百分比。
                            # iaa.Pad(percent=(0, 0.02)),  # 填充增强器,往四周填充一定比例的背景，四周填充的背景大小不一样。
                            # iaa.Affine(rotate=(-1, 1)),  # 以一定的角度旋转图像
                            # iaa.Affine(shear=(-1,1)), #以一定的角度错切图像,0~360度之间,正负表示方向
                            iaa.Affine(shear=(-1, 1)),  # 以一定的角度错切图像,0~360度之间,正负表示方向
                            iaa.Affine(translate_percent=(-0.01, 0.01)),  # 平移比例,0表示不平移,0.5表示平移50%。用正负来表示平移方向。
                            iaa.PiecewiseAffine(scale=(0.0, 0.01)),  # 扭曲图像增强器
                            # iaa.Dropout(p=0.01),
                            iaa.GaussianBlur((0.2, 0.5)),
                            iaa.Fliplr(1),  # 将所有图像中指定比例数量的图像进行左右翻转
                            # iaa.Flipud(1),  # 将所有图像中指定比例数量的图像进行上下翻转
                            ], random_order=True)
    ],
        random_order=True
    )
else:
    # bad
    seq = iaa.Sequential([iaa.SomeOf((4, 7), [
        # iaa.AddToHueAndSaturation((-3, 3)),  # 将图像转换至HSV空间，改变H和S值。
        iaa.Grayscale((0.0, 0.1), 'BGR'),  # 第一个参数：转换后的图像的可见度；第二个参数：原图像的读取方式。
        iaa.ContrastNormalization((0.8, 1.2)),  # 表示以0.8到1.2之间的随机因子来改变图像的对比度，参数值可修改。
        iaa.Add((-10, 10)),  # 为图像中所有像素增加值，此例中的值为-20到20之间。
        # iaa.Add((1, 20), per_channel=0.5), #为图像中所有像素增加值，此例中的值为-20到20之间。
        iaa.Multiply((0.8, 1.2)),  # 乘法增强器，将图像中所有像素乘以特定值，使图像更亮或者更暗，第二个参数表示作用的通道数百分比。
        # iaa.Pad(percent=(0, 0.02)),  # 填充增强器,往四周填充一定比例的背景，四周填充的背景大小不一样。
        iaa.Affine(rotate=(-5, 5)),  # 以一定的角度旋转图像
        # iaa.Crop(percent=(0, 0.2), keep_size=True),
        iaa.Affine(shear=(-1,1)), #以一定的角度错切图像,0~360度之间,正负表示方向
        iaa.Affine(shear=(-5, 5)),  # 以一定的角度错切图像,0~360度之间,正负表示方向
        iaa.Affine(translate_percent=(-0.01, 0.01)),  # 平移比例,0表示不平移,0.5表示平移50%。用正负来表示平移方向。
        iaa.PiecewiseAffine(scale=(0.0, 0.01)),  # 扭曲图像增强器
        # iaa.Dropout(p=0.005),
        iaa.GaussianBlur((0.2, 0.5)),
        iaa.Fliplr(1),  # 将所有图像中指定比例数量的图像进行左右翻转
        iaa.Flipud(1),  # 将所有图像中指定比例数量的图像进行上下翻转
        # iaa.CoarseDropout((0.0,0.01),size_percent=0.5)
    ], random_order=True)
                          ],
                         random_order=True
                         )

if __name__ == '__main__':

    root = r'D:\llf\dataset\danyang\2024_dataset\tmp'
    rootdir = os.path.join(root, 'input')
    augdir = os.path.join(root, 'aug')
    aug_rate = 5
    if os.path.exists(augdir):
        print(f'clean {augdir}')
        shutil.rmtree(augdir)
    os.makedirs(augdir)
    assert os.path.exists(rootdir), f'error:not this dir'
    assert os.path.exists(augdir), f'error:not this dir'

    for subdir, dirs, files in os.walk(rootdir):
        for file in tqdm(files):
            name = subdir + os.sep + file
            if 1:  # name.endswith(".png"):
                # print(name)
                images = cv2.imread(name)
                for idx in range(aug_rate):
                    images_aug = seq.augment_image(images)
                    # print(augdir + os.sep + file[:-4] + '_augment' + str(idx) + '.png')
                    cv2.imwrite(augdir + os.sep + file[:-4] + '_augment' + str(idx) + '.png', images_aug)
