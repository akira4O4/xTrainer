import os
import torch
import onnx
from onnxsim import simplify
from utils.util import get_time
from typing import Optional
from loguru import logger
import shutil
from network.shufflenetv2 import shufflenet_v2_x1_0
from network.shufflenetv2_segmantationplus_inference import shufflenet_v2_x1_0 as seg_shufflenet_v2_x1_0
from network.shufflenetv2_multi_taskplus_inference import shufflenet_v2_x1_0 as multi_task_shufflenet_v2_x1_0


def model_backup():
    ...


def export_onnx(
        model,
        img: torch.Tensor,
        opset_version: int,
        output_path: str,
        input_names: list,
        output_names: list,
        dynamic_axes: dict = None,
        is_simplify: bool = True,
        verbose=False,
        **kwargs
):
    logger.info(f'Starting ONNX export with onnx {onnx.__version__}...')
    output_path = output_path + ('_Static.onnx' if dynamic_axes is None else '_Dynamic.onnx')

    torch.onnx.export(
        model,
        img,
        output_path,
        verbose=verbose,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        **kwargs
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    if is_simplify:
        onnx_model, _ = simplify(onnx_model)
        onnx.save(onnx_model, output_path)

    logger.success('Export onnx model done.')


def model_prepare(model, model_path: str):
    checkpoint = torch.load(model_path, map_location='cpu')  # pytorch模型地址
    static_dict = checkpoint['state_dict']
    model.load_state_dict(static_dict, strict=False)

    model = model.cpu()
    model = model.eval()
    return model


def get_danyang_E_config() -> dict:
    return {
        'model_path': r"D:\llf\code\pytorch-lab\project\danyang_E_mt\20240422_171751_Epoch95_Acc99.9709_MIoU0.7555_lr1.2e-05_BestModel.pth",
        'nchw': [1, 3, 576, 576],
        'num_classes': 4,
        'mask_classes': 6,
        'onnx_model_name': '_danyang_E_',
        'input_names': ['images'],
        'output_names': ['output1', 'output2'],
        # 'model ': multi_task_shufflenet_v2_x1_0(num_classes=3, mask_classes=6),
    }


def get_danyang_G_config() -> dict:
    return {
        'model_path': r"D:\llf\code\pytorch-lab\project\danyang_E_mt\20240422_171751_Epoch95_Acc99.9709_MIoU0.7555_lr1.2e-05_BestModel.pth",
        'nchw': [1, 3, 480, 480],
        'num_classes': 2,
        'mask_classes': 0,
        'onnx_model_name': '_danyang_G_',
        'input_names': ['images'],
        'output_names': ['output'],
        'model': shufflenet_v2_x1_0(num_classes=1),
    }


def get_danyang_F_config() -> dict:
    return {
        'model_path': r"D:\llf\code\pytorch-lab\project\danyang_E_mt\20240422_171751_Epoch95_Acc99.9709_MIoU0.7555_lr1.2e-05_BestModel.pth",
        'nchw': [1, 3, 576, 576],
        'num_classes': 0,
        'mask_classes': 6,
        'onnx_model_name': '_danyang_F_',
        'input_names': ['images'],
        'output_names': ['output'],
        'model': seg_shufflenet_v2_x1_0(mask_classes=6),
    }


def get_danyang_C2_config() -> dict:
    return {
        'model_path': r"D:\llf\code\pytorch-lab\project\danyang_E_mt\20240422_171751_Epoch95_Acc99.9709_MIoU0.7555_lr1.2e-05_BestModel.pth",
        'nchw': [1, 3, 480, 479],
        'num_classes': 3,
        'mask_classes': 0,
        'onnx_model_name': '_danyang_E_',
        'input_names': ['images'],
        'output_names': ['output0'],
        'model': shufflenet_v2_x1_0(num_classes=3),
    }


if __name__ == '__main__':
    curr_time = get_time()
    model_str = '{}_{}_{}_{}_CLS{}_SEG{}'  # N_C_H_W_CLS_SEG.ONNX

    export_config = get_danyang_E_config()
    # export_config = get_danyang_G_config()
    # export_config = get_danyang_F_config()
    # export_config = get_danyang_C2_config()
    n = export_config['nchw'][0]
    c = export_config['nchw'][1]
    h = export_config['nchw'][2]
    w = export_config['nchw'][3]

    onnx_output_path = curr_time + \
                       export_config['onnx_model_name'] + \
                       model_str.format(n, c, h, w, export_config['num_classes'], export_config['mask_classes'])

    export_args = {
        'model': None,
        'img': None,
        'opset_version': 12,
        'output_path': os.path.join(f'../', onnx_output_path),
        'input_names': export_config['input_names'],
        'output_names': export_config['output_names'],
        # 'dynamic_axes': {
        #     'images': {0: 'batch'},
        #     'output1': {0: 'batch'},
        #     # 'output2': {0: 'batch'},
        # },
        'is_simplify': True
    }

    model = model_prepare(export_config['model'], export_config['model_path'])
    img = torch.zeros(export_config['nchw'])

    export_args['model'] = model
    export_args['img'] = img

    export_onnx(**export_args)
