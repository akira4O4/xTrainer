import os, json

import cv2
import numpy as np
import shutil
from tqdm import tqdm


def seekImage(root):
    pathList = []
    for i, j, k in os.walk(root):
        for l in k:
            if l.split('.')[-1] in {'json'}:
                path = os.path.join(i, l)
                pathList.append(path)
    return pathList


def json_loader(path):
    with open(path, 'rb') as f:
        jsondata = json.load(f)
    return jsondata


if __name__ == '__main__':
    root = r"C:\Users\Administrator\Desktop\tmp\w"
    # output = r""
    samples = seekImage(root)
    labeldic = {}
    cnt=0
    for path in tqdm(samples):
        if ".json" in path:
            datajson = json_loader(path)
            for i, roi in enumerate(datajson["shapes"]):
                if roi["label"] == '1_heidian':
                    datajson["shapes"][i]["label"] = "5_jipianshang"
                    cnt+=1
            with open(path, "w") as f:
                json.dump(datajson, f)
    print(cnt)
