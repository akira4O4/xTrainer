import os
import torch
import onnx
from onnxsim import simplify
from loguru import logger
from src.utils.util import get_time, load_yaml
import shutil
from src.network.shufflenetv2 import shufflenet_v2_x1_0 as classification_shufflenet
from src.network.shufflenetv2_segmantationplus_inference import shufflenet_v2_x1_0 as segmentation_shufflenet
from src.network.shufflenetv2_multi_taskplus_inference import shufflenet_v2_x1_0 as multi_task_shufflenet


def export(
    model,
    img: torch.Tensor,
    opset_version: int,
    output_path: str,
    input_names: list,
    output_names: list,
    dynamic_axes: dict = None,
    is_simplify: bool = True,
    verbose=True,
    **kwargs
):
    logger.info('Starting ONNX export with onnx %s...' % onnx.__version__)

    torch.onnx.export(
        model,
        img,
        output_path,
        verbose=verbose,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    if is_simplify:
        onnx_model, _ = simplify(onnx_model)
        onnx.save(onnx_model, output_path)

    logger.success('Export onnx model done.')


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


def main(args: dict) -> None:
    task = args['project_config']['task']
    export_config = args['onnx_export_config']

    if os.path.exists(export_config['model_path']) is False:
        logger.error('input model path is not found.')
        exit()

    model = model_prepare(
        task,
        export_config.get('num_classes'),
        export_config.get('mask_classes'),
        export_config.get('model_path')
    )
    export_config.update({'model': model})

    img = torch.zeros((
        export_config['batch_size'],
        export_config['channel'],
        export_config['input_h'],
        export_config['input_w']
    ))
    export_config.update({'img': img})

    file_name, file_ext = os.path.splitext(export_config['output_name'])  # [xxx,.onnx]
    output_name = file_name + ('_static.onnx' if export_config['is_dynamic'] is None else '_dynamic.onnx')

    export_time = get_time()
    output_dir = os.path.join('../', args['project_config']['work_dir'], 'export_onnx', export_time)

    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    export_config.update({
        'output_path': os.path.join(output_dir, output_name)
    })

    if export_config['is_dynamic'] is False:
        export_config['dynamic_axes'] = None

    shutil.copy(export_config['model_path'], output_dir)  # backup pytorch model
    export(**export_config)


if __name__ == '__main__':
    args = {
        'model_path': '',
        'fuse': True,
        'num_classes': 3,
        'mask_classes': 0,
        'output_name': 'classification.onnx',
        'batch_size': 1,
        'channel': 3,
        'input_h': 480,
        'input_w': 480,
    }
    # config_path = 'D:\llf\code\pytorch-lab\configs\default\classification.yml'
    main(args)
