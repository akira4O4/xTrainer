from utils.util import get_images, load_json
from tqdm import tqdm
import os
import json
import shutil

if __name__ == '__main__':
    root = r"D:\llf\dataset\danyang\B\B-labelme\dataset\B-crop-2"
    output = r'D:\llf\dataset\danyang\B\B-labelme\dataset\B-crop-2-clean'
    background_output = r'D:\llf\dataset\danyang\B\B-labelme\dataset\B-crop-2-background'
    remove_label_name = ['manyPoint', 'ngPoint']

    if not os.path.exists(output):
        os.makedirs(output)
        print(f'create:{output}')

    if not os.path.exists(background_output):
        os.makedirs(background_output)
        print(f'create:{background_output}')

    images = get_images(root)
    for image in tqdm(images):
        basename = os.path.basename(image)
        name, ext = os.path.splitext(basename)
        json_file = image.replace(ext, '.json')

        new_shapes = []
        if os.path.exists(json_file):
            json_data = load_json(json_file)
            shapes = json_data['shapes']

            for shape in shapes:

                if shape['label'] not in remove_label_name:
                    new_shapes.append(shape)

            if len(new_shapes) == 0:
                shutil.copy(image, os.path.join(background_output, basename))

            else:
                json_data['shapes'] = new_shapes

                new_json_path = os.path.join(output, os.path.basename(json_file))
                new_img_path = os.path.join(output, basename)

                shutil.copy(image, new_img_path)
                with open(new_json_path, "w") as f:
                    json.dump(json_data, f)
