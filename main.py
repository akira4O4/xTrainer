import os
import shutil
from exp.pipline import Pipline

if __name__ == '__main__':
    remove_path=r'D:\llf\code\pytorch-lab\project\test_mt\runs'
    if os.path.exists(remove_path):
        shutil.rmtree(remove_path)

    config_path = r'D:\llf\code\pytorch-lab\configs\test_mt.yml'
    pipline = Pipline(config_path)
    pipline.run()
