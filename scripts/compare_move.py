import os
import shutil


def find_image(root: str) -> list:
    output = []
    ext = ['jpg', 'png', 'jpeg']
    for file in os.listdir(root):
        if not os.path.isfile(file):
            continue
        file_extend = file.split(".")[-1]
        if file_extend in ext:
            output.append(file)

    # for root, dirs, files in os.walk(root):
    #     for file in files:
    #         file_extend = file.split(".")[-1]
    #         if file_extend in ext:
    #             output.append(os.path.join(root, file))
    #             output.append(file)
    return output


def compare_move(root: str, target: str, output: str):
    root_images = find_image(root)
    target_images = find_image(target)
    # print(root_images)
    for image in target_images:
        if image in root_images:
            # shutil.move(os.path.join(root, image), os.path.join(output, image))
            print(f'move {os.path.join(root, image)} to {os.path.join(output, image)}')


if __name__ == '__main__':
    compare_move(root='/home/lee/Desktop/aaa',
                 target='/home/lee/Desktop/bbb',
                 output='/home/lee/Desktop/ccc')
