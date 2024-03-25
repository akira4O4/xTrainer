import sys

sys.path.append("../../../../")

import torch
import torch.nn as nn
from torchvision import transforms

from network.shufflenetv2_multi_taskplus_inference import shufflenet_v2_x1_0


# 参考torch.quantization.fuse_modules()
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens[:-1]:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def _del_module(model, submodule_key):
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens[:-1]:
        cur_mod = getattr(cur_mod, s)
    # delattr(cur_mod, tokens[-1])
    setattr(cur_mod, tokens[-1], nn.Identity())


def fuse(conv, bn):
    fused = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        dilation=conv.dilation,
        bias=True
    )

    # setting weights
    # w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_conv = conv.weight.clone().view(conv.weight.shape[0], -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.size()))

    # setting bias
    if conv.bias is not None:
        b_conv = conv.bias.mul(bn.weight).div(
            torch.sqrt(bn.running_var + bn.eps)
        )
    else:
        b_conv = torch.zeros(conv.weight.size(0))
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fused.bias.copy_(b_conv + b_bn)

    return fused


batch_size = 1

# hr 1 front
# cls = 11
# seg_cls = 5
# w = 352
# h = 352
# model_path = r"/home/llf/PycharmProjects/deep_learn_frame_v2/project/huiren_1_front/weights/checkpoint.pth"
# output_prefix = f'../torchscipt/20230403/hr_#1_front_'
# output_suffix = f'cls{cls}_segcls{seg_cls}_{w}x{h}.libtorch.pt'

# hr 1 back
cls = 4
seg_cls = 3
w, h = 352, 352
model_path = r"D:\llf\code\deep_learn_frame_v2\project\huiren_1_back\weights\checkpoint.pth"  # side
output_prefix = f'../torchscript_output/hr_#1_back_'
output_suffix = f'cls{cls}_segcls{seg_cls}_{w}x{h}.libtorch.pt'

# hr 2 front
# cls = 6
# seg_cls = 2
# w = 288
# h = 288
# model_path = r"/home/llf/PycharmProjects/deep_learn_frame_v2/project/huiren_2_front/weights/checkpoint.pth"  # side
# output_prefix = f'../torchscipt/20230403/hr_#2_front_'
# output_suffix = f'cls{cls}_segcls{seg_cls}_{w}x{h}.libtorch.pt'

# hr 2 back
cls = 3
seg_cls = 5
w = 288
h = 288
model_path = r"/home/llf/PycharmProjects/deep_learn_frame_v2/project/huiren_2_back/weights/checkpoint.pth"  # side
output_prefix = f'../torchscipt_output/20230403/hr_#2_back_'

output_suffix = f'cls{cls}_segcls{seg_cls}_{w}x{h}.libtorch.pt'
output_path = output_prefix + output_suffix

# preprocess
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)

transforms_ = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    normalize,
])

# model
model = shufflenet_v2_x1_0(num_classes=cls, mask_classes=seg_cls)

checkpoint = torch.load(model_path)  # pytorch模型地址
static_dict = checkpoint['state_dict']
model.load_state_dict(static_dict, strict=False)

rand_input = torch.rand(batch_size, 3, h, w)

torch.set_grad_enabled(False)

model.eval()
if True:
    y1 = model(rand_input)
    #
    layer_pre = None
    # print(model)
    for i, (name, layer) in enumerate(list(model.named_modules())):
        if i > 0:
            if isinstance(layer_pre, nn.Conv2d):
                # print(layer_pre.kernel_size)
                # if len(layer_pre.kernel_size) == 2:
                #     if (layer_pre.kernel_size[0] != 3) or (layer_pre.kernel_size[1] != 3):
                #         layer_pre = layer
                #         continue
                if isinstance(layer, nn.BatchNorm2d):
                    newconv = fuse(layer_pre, layer)
                    _set_module(model, name_pre, newconv)
                    _del_module(model, name)

        name_pre = name
        layer_pre = layer
    #
    # print(model)
    y2 = model(rand_input)
    print(abs(y1[0] - y2[0]).sum())
    print(abs(y1[1] - y2[1]).sum())

model.cuda()

rand_input = torch.rand(batch_size, 3, h, w).cuda()

model_torchscript = torch.jit.trace(model, rand_input, strict=False)
model_torchscript.save(output_path)
print(f'export torchscript done. path:{output_path}')
