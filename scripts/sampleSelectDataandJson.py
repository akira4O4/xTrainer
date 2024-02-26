'''
Descripttion: 
version: 
Author: ZaoShan
Date: 2022-04-10 23:25:03
LastEditors: Zaoshan
LastEditTime: 2022-04-10 23:30:49
'''
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:11:22 2019

@author: Administrator
"""

import random, shutil, os


def seekImage(imageRoot):
    imagePathList = []
    for i, j, k in os.walk(imageRoot):
        for l in k:
            if l.split('.')[-1] in {'jpg', 'jpeg', 'bmp', 'png'}:
                path = os.path.join(i, l)
                imagePathList.append(path)
    return imagePathList


imageRoot = r'/home/lee/PycharmProjects/deep_learn_frame/project/shenghuofront/temp/output/classeserror/3zhezhou/118'  # 这是你将要c进行划分的数据目录，后面可以有很多的子文件夹，子文件夹也可以嵌套，但是你这里只需要修改成你的目录，子文件夹不需要管它
saveRoot = r'/home/lee/PycharmProjects/deep_learn_frame/project/shenghuofront/temp/output/classeserror/3zhezhou/output'  # 这是你要用来存储数据的目录，子文件夹通过后面的代码自动生成，不需要自己设置
imagePathList = seekImage(imageRoot)  # 得到所有图片的路径
# sampleImageList=random.sample(imagePathList,4000) #通过设定的数量从imagePathList列表中随机选择元素，形成新路径列表sampleImageList
img_dict = {}
for imagePath in imagePathList:  # 遍历随机选择的图片路径2
    if os.path.exists(os.path.splitext(imagePath)[0] + ".json"):
        img_dict[imagePath] = os.path.splitext(imagePath)[0] + ".json"
    else:
        img_dict[imagePath] = None

selectcount = int(len(imagePathList) * 0.3)
sampleImageList = random.sample(imagePathList, selectcount)  # 或者将数量设置成一定比例，比如0.2，这个比例可以自己修改
print(len(sampleImageList), len(imagePathList) - len(sampleImageList))
for imagePath in sampleImageList:  # 遍历随机选择的图片路径2
    dstPath = saveRoot + imagePath.split(imageRoot)[-1]
    # labelWords = imageRoot.split('\\')[-1] # 得到标签关键字，这里为将'E:\a\good'处理得到'good'
    # tailString = imagePath.split(labelWords)[-1] #将图片路径根据标签关键字进行分割，取分割后的最后一个元素，得到标签关键字之后的子路径
    # basename = os.path.basename(imagePath) #得到图片名
    # midString = tailString.strip(basename) # 将tailString去掉basename，得到没有basename的新字符串
    # dstPath = saveRoot+'\\'+labelWords+midString # 得到你每张图片对应需要保存的最终目录
    # os.makedirs(dstPath,exist_ok=True) # 生成目录
    aa = dstPath.split(os.path.basename(dstPath))
    os.makedirs(dstPath.split(os.path.basename(dstPath))[0], exist_ok=True)
    shutil.copy(imagePath, dstPath)  # 将随机选出的每张图片存储到它对应的目录下，文件夹层级将保持不变。
    if img_dict[imagePath]:
        shutil.copy(img_dict[imagePath], os.path.splitext(dstPath)[0] + ".json")
