import os
import numpy as np
import cv2

if __name__ == '__main__':
    n = 10
    c = 3
    h = 224
    w = 224
    suffix = 'jpg'
    output = r'D:\llf\code\pytorch-lab\temp'

    for i in range(n):
        dummy = np.random.normal(size=(h, w, c))
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(output, str(i) + f'.{suffix}'), dummy)
