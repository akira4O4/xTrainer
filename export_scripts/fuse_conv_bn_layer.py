import os
import copy
import time
from typing import Optional

import torch
import torch.nn as nn
import torchvision


class FuseConvBN:
    def __init__(self, model, input_data: torch.Tensor):

        torch.set_grad_enabled(False)

        self._model = model
        self._fuse_model = copy.deepcopy(model)

        self.infer_times = 32
        self.input_data = input_data
        self.input_data.cuda()

        self._model_output = None
        self._fuse_model_output = None

    @property
    def model_output(self) -> torch.Tensor:
        return self._model_output

    @property
    def fuse_model_output(self) -> torch.Tensor:
        return self._fuse_model_output

    @property
    def fuse_model(self) -> nn.Module:
        return self._fuse_model

    @staticmethod
    def _replace_layer(model: nn.Module, key: str, layer: nn.Module) -> None:
        setattr(model, key, layer)

    def fuse(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Module:  # noqa
        fused_layer = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,  # noqa
            stride=conv.stride,  # noqa
            padding=conv.padding,
            bias=True
        )

        """
        x^  =   wx+b
        w   =   γ/sqrt(σ²+ε)
        b   =   β-γμ/sqrt(σ²+ε)
        wx+b = (γ/sqrt(σ²+ε))x + β-γμ/sqrt(σ²+ε) 
        wx+b = (γ/sqrt(σ²+ε))x_w + (γ/sqrt(σ²+ε))x_b + β-γμ/sqrt(σ²+ε) 
        """
        bn_gamma = bn.weight
        bn_beta = bn.bias

        # Weight fuse
        conv_weight = conv.weight.clone().view(conv.out_channels, -1)
        bn_weight = torch.diag(bn_gamma.div(torch.sqrt(bn.running_var + bn.eps)))
        fused_layer.weight.copy_(torch.mm(bn_weight, conv_weight).view(fused_layer.weight.size()))

        # Bias fuse
        if conv.bias is not None:
            conv_bias = conv.bias
        else:
            conv_bias = torch.zeros(conv.weight.size(0))

        conv_bias = torch.mm(bn_weight, conv_bias.view(-1, 1)).view(-1)

        bn_bias = bn_beta - bn_gamma.mul(bn.running_mean).div(
            torch.sqrt(bn.running_var + bn.eps)
        )
        fused_layer.bias.copy_(conv_bias + bn_bias)

        return fused_layer

    def warmup(self, model: nn.Module, times: Optional[int] = 10) -> None:
        for _ in range(times):
            __ = model(self.input_data)

    def model_infer_time(self) -> None:
        self._model.cuda()

        self.warmup(self._model)

        t1 = time.time()
        for _ in range(self.infer_times):
            self._model_output = self._model(self.input_data)
        t2 = time.time()

        spend_time = (t2 - t1) / self.infer_times
        print(f'Model Infer Avg Time: {round(spend_time, 4)}s')

    def fuse_model_infer_time(self) -> None:
        self._fuse_model.cuda()

        self.warmup(self._fuse_model)
        t1 = time.time()
        for _ in range(self.infer_times):
            self._fuse_model_output = self._fuse_model(self.input_data)
        t2 = time.time()
        spend_time = (t2 - t1) / self.infer_times

        print(f'Fuse Model Infer Avg Time: {round(spend_time, 4)}s')

    def model_error(self):
        self._model.cuda()
        self._fuse_model.cuda()
        self._model_output = self._model(self.input_data)
        self._fuse_model_output = self._fuse_model(self.input_data)
        print(f'Model Error: {(self._model_output - self._fuse_model_output).mean().item()}')

    def begin_convert(self) -> None:
        self._fuse_model.cpu()
        layer_pre = None
        name_pre = None
        for i, (name, layer) in enumerate(list(self._fuse_model.named_modules())):
            if i > 0:
                if isinstance(layer_pre, nn.Conv2d):
                    if isinstance(layer, nn.BatchNorm2d):
                        fuse_layer = self.fuse(layer_pre, layer)
                        self._replace_layer(self._fuse_model, name_pre, fuse_layer)
                        self._replace_layer(self._fuse_model, name, nn.Identity())

            name_pre = name
            layer_pre = layer

        self.model_error()

    def save_fuse_model(self, save_path: Optional[str] = './') -> None:
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        model_save_path = os.path.join(save_path, 'fuse_model.pth')
        torch.save(self._fuse_model, model_save_path)
        print(f'Save Fuse Model: {model_save_path}')


if __name__ == '__main__':
    rand_input = torch.rand(4, 3, 416, 416).cuda()
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    fuse_conv_bn = FuseConvBN(model, rand_input)

    fuse_conv_bn.model_infer_time()
    fuse_conv_bn.begin_convert()
    fuse_conv_bn.fuse_model_infer_time()

    fuse_conv_bn.save_fuse_model()
