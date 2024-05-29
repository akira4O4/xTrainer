import os
import torch
import onnx
from onnxsim import simplify
from utils.util import get_time
from loguru import logger
import shutil
from network.shufflenetv2 import shufflenet_v2_x1_0
from network.shufflenetv2_segmantationplus_inference import shufflenet_v2_x1_0 as seg_shufflenet_v2_x1_0
from network.shufflenetv2_multi_taskplus_inference import shufflenet_v2_x1_0 as multi_task_shufflenet_v2_x1_0


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
        # **kwargs
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


danyang_E_config = {
    'opset_version': 12,
    'model_path': r"D:\llf\code\pytorch-lab\project\danyang_E_mt\20240509_104153_Epoch99_Acc99.965_MIoU0.7467_lr1.2e-05_BestModel.pth",
    'model': multi_task_shufflenet_v2_x1_0(num_classes=4, mask_classes=6),
    'img': None,
    'nchw': [1, 3, 576, 576],
    'num_classes': 4,
    'mask_classes': 6,
    'onnx_model_name': '_danyang_E_',
    'input_names': ['images'],
    'output_names': ['output1', 'output2'],
    'is_simplify': True,
    # 'dynamic_axes': {
    #     'images': {0: 'batch'},
    #     'output1': {0: 'batch'},
    #     # 'output2': {0: 'batch'},
    # }
}

danyang_G_config = {
    'opset_version': 12,
    'model_path': r"D:\llf\code\pytorch-lab\project\danyang_G\weights\20240315_015742_Epoch97_Acc100.0_lr7.4e-05_BestModel.pth",
    'model': shufflenet_v2_x1_0(num_classes=2),
    'img': None,
    'nchw': [1, 3, 480, 480],
    'num_classes': 2,
    'mask_classes': 0,
    'onnx_model_name': '_danyang_G_',
    'input_names': ['images'],
    'output_names': ['output'],
    'is_simplify': True,
    # 'dynamic_axes': {
    #     'images': {0: 'batch'},
    #     'output1': {0: 'batch'},
    #     # 'output2': {0: 'batch'},
    # }
}

danyang_F_config = {
    'opset_version': 12,
    'model_path': r"D:\llf\code\pytorch-lab\project\danyang_F_seg_exp2\weights\20240515_213213_Epoch199_MIoU0.9518_lr1e-05_BestModel.pth",
    'model': seg_shufflenet_v2_x1_0(mask_classes=6),
    'img': None,
    'nchw': [1, 3, 256, 256],
    'num_classes': 0,
    'mask_classes': 6,
    'onnx_model_name': '_danyang_F_',
    'input_names': ['images'],
    'output_names': ['output'],
    'is_simplify': True,
    # 'dynamic_axes': {
    #     'images': {0: 'batch'},
    #     'output1': {0: 'batch'},
    #     # 'output2': {0: 'batch'},
    # }
}

danyang_C2_config = {
    'opset_version': 12,
    'model_path': r"D:\llf\code\pytorch-lab\project\danyang_C2\20240121_004244_Epoch49_Acc98.4857_lr8e-06_BestModel.pth",
    'model': shufflenet_v2_x1_0(num_classes=3),
    'img': None,
    'nchw': [1, 3, 480, 480],
    'num_classes': 3,
    'mask_classes': 0,
    'onnx_model_name': '_danyang_C2_',
    'input_names': ['images'],
    'output_names': ['output'],
    'is_simplify': True,
    # 'dynamic_axes': {
    #     'images': {0: 'batch'},
    #     'output1': {0: 'batch'},
    #     # 'output2': {0: 'batch'},
    # },
}

if __name__ == '__main__':
    curr_time = get_time()
    onnx_output_path = f'../onnx_output/{curr_time}'
    if not os.path.exists(onnx_output_path):
        os.makedirs(onnx_output_path)

    model_str = '{}_{}_{}_{}_CLS{}_SEG{}'  # N_C_H_W_CLS_SEG.ONNX

    # export_config = danyang_E_config
    export_config = danyang_G_config
    # export_config = danyang_F_config
    # export_config = danyang_C2_config
    #
    onnx_output_name = curr_time + export_config['onnx_model_name'] + model_str.format(
        export_config['nchw'][0],
        export_config['nchw'][1],
        export_config['nchw'][2],
        export_config['nchw'][3],
        export_config['num_classes'],
        export_config['mask_classes']
    )

    shutil.copy(export_config['model_path'], onnx_output_path)  # backup pytorch model
    export_config['output_path'] = os.path.join(onnx_output_path, onnx_output_name)

    model = model_prepare(export_config['model'], export_config['model_path'])
    img = torch.zeros(export_config['nchw'])

    export_config['model'] = model
    export_config['img'] = img

    export_onnx(**export_config)
