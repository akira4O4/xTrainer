import os
import shutil
from exp.trainer import Trainer

if __name__ == '__main__':
    # remove_path = r'D:\llf\code\pytorch-lab\project\test_mt\runs'
    # if os.path.exists(remove_path):
    #     shutil.rmtree(remove_path)

    config_path = r'D:\llf\code\pytorch-lab\configs\default\classification.yml'
    trainer = Trainer(config_path)
    trainer.run()
