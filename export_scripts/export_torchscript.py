import os
import shutil

import torch
from loguru import logger

from src.args import TorchScriptArgs
from utils.util import load_yaml, get_time
from fuse_conv_bn_layer import FuseConvBN

from network.shufflenetv2 import shufflenet_v2_x1_0 as classification_shufflenet
from network.shufflenetv2_segmantationplus_inference import shufflenet_v2_x1_0 as segmentation_shufflenet
from network.shufflenetv2_multi_taskplus_inference import shufflenet_v2_x1_0 as multi_task_shufflenet


def model_prepare(task: str, num_classes: int, mask_classes: int, model_path: str):
    model = None
    if task == 'classification':
        model = classification_shufflenet(num_classes=num_classes)
    elif task == 'segmentation':
        model = segmentation_shufflenet(mask_classes=mask_classes)
    elif task == 'multi_task':
        model = multi_task_shufflenet(
            num_classes=num_classes,
            mask_classes=mask_classes
        )

    checkpoint = torch.load(model_path, map_location='cpu')
    static_dict = checkpoint['state_dict']
    model.load_state_dict(static_dict, strict=False)

    model = model.cpu()
    model = model.eval()
    return model


def export(model, input_data: torch.Tensor, is_fuse: bool, output_path: str) -> None:
    if is_fuse:
        fuse_conv_bn = FuseConvBN(model, input_data)
        fuse_conv_bn.begin_convert()
        model = fuse_conv_bn.fuse_model

    model.cuda()
    rand_input = input_data.cuda()
    model_torchscript = torch.jit.trace(model, rand_input, strict=False)
    model_torchscript.save(output_path)
    print(f'Export torchscript done. path:{output_path}')


def main(config_path: str):
    config = load_yaml(config_path)
    task = config['project_config']['task']
    export_config = TorchScriptArgs(**config['export_torchscript_config'])

    if os.path.exists(export_config.model_path) is False:
        logger.error('input model path is not found.')
        exit()

    model = model_prepare(
        task,
        export_config.num_classes,
        export_config.mask_classes,
        export_config.model_path
    )
    export_config.model = model

    rand_input = torch.rand(
        export_config.batch_size,
        export_config.channel,
        export_config.input_h,
        export_config.input_w
    )
    torch.set_grad_enabled(False)
    model.eval()

    export_time = get_time()
    output_dir = os.path.join('../', config['project_config']['work_dir'], 'export_torchscript', export_time)

    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, export_config.output_name)

    export(model, rand_input, export_config.fuse, output_path)
    shutil.copy(export_config.model_path, output_dir)


if __name__ == '__main__':
    config_path = 'D:\llf\code\pytorch-lab\configs\default\classification.yml'
    main(config_path)
