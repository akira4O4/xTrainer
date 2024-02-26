import os, shutil

from tqdm import tqdm


def getdirs(dir):
    image_files, img_exts = [], ['.png', '.jpg', '.bmp']
    for sub_root, _, files in os.walk(dir):
        image_files += [os.path.join(sub_root, f) for f in files if os.path.splitext(f)[-1].lower() in img_exts]
    return image_files


def mode_data(keydir, deldir, copydir):
    keyimage_files = getdirs(keydir)
    delimage_files = getdirs(deldir)
    keylist = []
    for imp in tqdm(keyimage_files):
        keylist.append(os.path.basename(imp))

    for imp in tqdm(delimage_files):
        if os.path.basename(imp) in keylist:
            dst_imp = copydir + "\\" + os.path.basename(imp)
            shutil.move(imp, dst_imp)


if __name__ == '__main__':
    deldir = r'D:\data\0danyang\20240122\bad\E\seg\0'  # 刪除圖像路徑

    keydir = r'D:\data\0danyang\20240122\bad\E\seg\bad'  # 查詢圖像名字路徑

    output_dir = r'D:\data\0danyang\20240122\bad\E\seg\rep'  # copy出來圖像路徑
    # flag = 'BOARD3'
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    mode_data(keydir, deldir, output_dir)
