import os
import torch
import onnx
from onnxsim import simplify
from utils.util import get_time

from network.shufflenetv2 import shufflenet_v2_x1_0
from network.shufflenetv2_segmantationplus_inference import shufflenet_v2_x1_0 as seg_shufflenet_v2_x1_0
from network.shufflenetv2_multi_taskplus_inference import shufflenet_v2_x1_0 as multi_task_shufflenet_v2_x1_0


def export_onnx(model,
                opset_version: int,
                output_path: str,
                input_names: list,
                output_names: list,
                batch_size: int,
                channel: int,
                input_h: int,
                input_w: int,
                dynamic_axes: dict = None,
                is_simplify: bool = True,
                **kwargs):
    img = torch.zeros((batch_size, channel, input_h, input_w))
    model = model.cpu()
    model = model.eval()
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)

    file_name, file_ext = os.path.splitext(output_path)
    output_path = file_name + ('_static.onnx' if dynamic_axes is None else '_dynamic.onnx')

    torch.onnx.export(model,
                      img,
                      output_path,
                      verbose=False,
                      opset_version=opset_version,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)

    onnx_model = onnx.load(output_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    if is_simplify:
        onnx_model, _ = simplify(onnx_model)
        onnx.save(onnx_model, output_path)
    print('export onnx model done.')


if __name__ == '__main__':
    curr_time = get_time()
    # E
    # model_path = r"/\project\danyang_E_mt\weights\20240122_233416_Epoch21_Acc99.8881_MIoU0.7611_lr1.3e-05_BestModel.pth"
    # bs = 1
    # wh = [576, 576]
    # cls = 4
    # seg_cls = 6
    # onnx_name = f'{curr_time}_danyang_E_mt_bs{bs}_cls{cls}_seg{seg_cls}.onnx'
    # output_names = ['output1', 'output2']
    # model = multi_task_shufflenet_v2_x1_0(num_classes=cls, mask_classes=seg_cls)

    # G
    model_path = r"D:\llf\code\pytorch-lab\project\danyang_G\weights\20240316_030113_Epoch95_Acc100.0_lr4e-06_BestModel.pth"
    bs = 1
    wh = [480, 480]
    cls = 2
    onnx_name = f'{curr_time}_danyang_G_cls{cls}_bs{bs}.onnx'
    output_names = ['output1']
    model = shufflenet_v2_x1_0(num_classes=cls)

    # gz sapa xiansao
    # model_path = r"D:\code\DLFv2\project\gz_sapa_xiansao\weights\checkpoint.pth"
    # bs = 1
    # wh = [416, 416]
    # cls = 6
    # seg_cls=4
    # onnx_name = f'{curr_time}_gz_sapa_bs{bs}.onnx'
    # output_names = ['output1']
    # model = multi_task_shufflenet_v2_x1_0(num_classes=cls,mask_classes=seg_cls)

    # F
    # model_path = r"D:\code\DLFv2\project\danyang_F_seg_exp2\weights\20240122_215705_Epoch44_MIoU0.9433_lr9e-06_BestModel.pth"
    # bs = 1
    # wh = [256, 256]
    # seg_cls = 6
    # onnx_name = f'{curr_time}_danyang_F_seg_exp2_bs{bs}.onnx'
    # output_names = ['output1']
    # model = seg_shufflenet_v2_x1_0(mask_classes=seg_cls)

    # C2
    # model_path = r"D:\llf\code\pytorch-lab\project\danyang_C2\20240121_000636_Epoch44_Acc99.7782_lr9e-06_BestModel.pth"
    # bs = 1
    # wh = [480, 480]
    # cls = 3
    # onnx_name = f'{curr_time}_danyang_C2_cls_bs{bs}.onnx'
    # onnx_name = f'{curr_time}_danyang_C2_cls{cls}_bs{bs}.onnx'
    # output_names = ['output1']
    # model = shufflenet_v2_x1_0(num_classes=cls)

    export_args = {
        'opset_version': 12,
        'output_path': os.path.join(f'../', onnx_name),
        'input_names': ['images'],

        'output_names': output_names,
        'batch_size': bs,
        'channel': 3,

        'input_h': wh[1],
        'input_w': wh[0],

        # 'dynamic_axes': {
        #     'images': {0: 'batch'},
        #     'output1': {0: 'batch'},
        #     'output2': {0: 'batch'},
        # },
        'is_simplify': True
    }

    # cls = 2
    # seg_cls = 2
    # model = shufflenet_v2_x1_0(num_classes=cls)
    # model = seg_shufflenet_v2_x1_0(num_classes=cls, mask_classes=seg_cls)
    # model = multi_task_shufflenet_v2_x1_0(num_classes=cls, mask_classes=seg_cls)
    checkpoint = torch.load(model_path, map_location='cpu')  # pytorch模型地址
    static_dict = checkpoint['state_dict']
    model.load_state_dict(static_dict, strict=False)
    export_onnx(model, **export_args)
