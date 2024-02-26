import os
from sklearn import model_selection
import shutil
from tqdm import tqdm


def get_images(path, ext=None):
    if ext is None:
        ext = ["jpg", "png", "jpeg"]

    ret = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_extend = file.split(".")[-1]
            if file_extend in ext:
                ret.append(os.path.join(root, file))
    return ret


def get_test(dataset_path: str,
             save_path: str,
             test_ratio: float = 0.2,
             mode: str = 'copy',
             ):
    if mode not in ['copy', 'move']:
        print('mode must be copy or move')
        exit()
    print(f'Test ratio :{test_ratio}')
    images = get_images(dataset_path)
    print(f'Total image size:{len(images)}')

    sample_train, sample_test = model_selection.train_test_split(images, test_size=test_ratio)
    print(f'Train data size: {len(sample_train)}')
    print(f'Test data size: {len(sample_test)}')

    test_dir = os.path.join(save_path, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f'New test dir:{test_dir}')

    pbar = tqdm(sample_test, total=len(sample_test))
    for file in pbar:
        file_name = os.path.basename(file)
        new_file = os.path.join(test_dir, file_name)

        if os.path.exists(new_file):
            print(f'{new_file} is already exists -> skip')
            continue

        if mode == 'copy':
            shutil.copy(file, new_file)
        elif mode == 'move':
            shutil.move(file, new_file)

    print('done.')


if __name__ == '__main__':
    get_test('/home/lee/data/shenghuoside/train/6shuangli', save_path='/home/lee/data/_test/6shuangli')
