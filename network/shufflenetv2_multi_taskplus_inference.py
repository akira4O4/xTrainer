import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Callable, Any, List
import math
from utils.baisc import SPPF, MultiSampleDropout

__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int
    ) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
            i: int,
            o: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            # x2 = self.branch2(x2)
            # x2 = torch.squeeze(self.branch3(torch.unsqueeze(x2, dim=1)), dim=1)
            # out = torch.cat((x1, x2), dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            # x2 = self.branch2(x)
            # x2 = torch.squeeze(self.branch3(torch.unsqueeze(x2, dim=1)), dim=1)
            # out = torch.cat((self.branch1(x), x2), dim=1)
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class DownUpBone(nn.Module):
    def __init__(self, inplanes, out_channel):
        super(DownUpBone, self).__init__()
        # 使用平均池化，然后进行bn、relu、再进行一次,group conv 3 1操作
        # 使用平均池化功能类似于卷积功能，相当于模拟卷积操作。增大感受野
        out_planes = inplanes // 2

        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    nn.BatchNorm2d(inplanes),
                                    # nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, out_planes, kernel_size=(1, 1), bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    nn.BatchNorm2d(inplanes),
                                    # nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, out_planes, kernel_size=(1, 1), bias=False),
                                    )
        self.scale3 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=(3, 3), groups=inplanes, stride=(2, 2), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, out_planes, kernel_size=(1, 1), bias=False),
        )
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, out_planes, kernel_size=(1, 1), bias=False),
        )
        self.process1 = nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=False),
        )
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=False),
        )
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=False),
        )
        self.process4 = nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=False),
        )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(out_planes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes * 4, inplanes, kernel_size=(1, 1), bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, out_channel, kernel_size=(1, 1), bias=False),
        )

    def forward(self, x):
        x_list = []
        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x), scale_factor=2) + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x), scale_factor=4) + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x), scale_factor=2) + x_list[2])))
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class MaskBranch(nn.Module):
    def __init__(self, concat_ch=24, mask_classes=1):
        super(MaskBranch, self).__init__()
        # 定义每个阶梯的特征融合节点,使用固定的通道数。定义顺序从网络底部到上面

        self.conv11 = nn.Conv2d(concat_ch + 24, concat_ch, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn11 = nn.BatchNorm2d(concat_ch)
        self.relu11 = nn.ReLU(inplace=True)

        # self.conv21 = nn.Conv2d(256 + 128, concat_ch, kernel_size=(3, 3), padding=(1, 1), bias=False)
        # self.bn21 = nn.BatchNorm2d(concat_ch)
        # self.relu21 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(24 + 116 + concat_ch, concat_ch, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn12 = nn.BatchNorm2d(concat_ch)
        self.relu12 = nn.ReLU(inplace=True)

        # self.conv31 = nn.Conv2d(1024 + 232, concat_ch, kernel_size=(3, 3), padding=(1, 1), bias=False)
        # self.bn31 = nn.BatchNorm2d(concat_ch)
        # self.relu31 = nn.ReLU(inplace=True)

        self.conv22 = nn.Conv2d(116 + 232 + concat_ch, 116, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn22 = nn.BatchNorm2d(116)
        self.relu22 = nn.ReLU(inplace=True)

        # 网络输出
        self.process1 = nn.Sequential(
            nn.Conv2d(concat_ch, concat_ch, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(concat_ch),
            nn.ReLU(inplace=True),
        )
        # self.conv_01 = nn.Conv2d(concat_ch, mask_classes, kernel_size=(3, 3), padding=(1, 1))
        self.conv_01_1 = nn.Conv2d(concat_ch, mask_classes, kernel_size=(1, 1))

        self.process2 = nn.Sequential(
            nn.Conv2d(concat_ch, concat_ch, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(concat_ch),
            nn.ReLU(inplace=True),
        )
        # self.conv_02 = nn.Conv2d(concat_ch * 2, mask_classes, kernel_size=(3, 3), padding=(1, 1))
        self.conv_02_1 = nn.Conv2d(concat_ch * 2, mask_classes, kernel_size=(1, 1))

        # 定义upsample
        self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upscore4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.SPPF3 = SPPF(232, 232)
        self.SPPF4 = SPPF(1024, 1024)

        # self.dropout2d_1 = nn.Dropout2d(p=0.2)
        # self.dropout2d_2 = nn.Dropout2d(p=0.5)

        conv1 = nn.Sequential(
            nn.Conv2d(232 + 116, concat_ch, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(concat_ch),
            nn.ReLU(inplace=True),
        )
        conv2 = nn.Sequential(
            nn.Conv2d(1024 + 232, 232, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(232),
            nn.ReLU(inplace=True),
        )

        self.mutisampledropout2d_1 = MultiSampleDropout(conv1, dim=2, p=0.2)
        self.mutisampledropout2d_2 = MultiSampleDropout(conv2, dim=2, p=0.5)

        self.conv21_1 = nn.Conv2d(concat_ch, mask_classes, kernel_size=(1, 1))
        # self.conv11_1 = nn.Conv2d(concat_ch, mask_classes, kernel_size=(1, 1))
        self.conv22_1 = nn.Conv2d(116, mask_classes, kernel_size=(1, 1))
        # self.conv12_1 = nn.Conv2d(concat_ch, mask_classes, kernel_size=(1, 1))

        # def conv_dw(self,inp,out_channel):
        #     return nn.Sequential(
        #         nn.Conv2d(inp,inp,groups=inp,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
        #         nn.BatchNorm2d(inp),
        #         nn.Conv2d(inp,out_channel,kernel_size=(1,1),stride=(1,1)),
        #         nn.BatchNorm2d(out_channel),
        #         nn.ReLU(inplace=True),
        #     )
        #
        # def conv_131(self,inp):
        #     return nn.Sequential(
        #         nn.Conv2d(inp, inp//2, kernel_size=(1,1)),
        #         nn.Conv2d(inp//2,inp//2,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
        #         nn.BatchNorm2d(inp//2),
        #         nn.Conv2d(inp//2,inp,kernel_size=(1,1)),
        #         nn.BatchNorm2d(inp),
        #         nn.ReLU(inplace=True)
        #     )

    def forward(self, x1, x2, x3, x4):  #
        """
        Args:
            x1: 网络底部的输出特征
            x2:
            x3:
            x4:

        Returns:

        """
        x3 = self.SPPF3(x3)
        x_up = self.upscore2(x3)
        x2_1 = torch.cat([x_up, x2], dim=1)

        # x2_1 = self.dropout2d_1(x2_1)
        # x2_1 = self.relu21(self.bn21(self.conv21(x2_1)))
        x2_1 = self.mutisampledropout2d_1(x2_1)

        x2_1_out = self.conv21_1(x2_1)

        x_up = self.upscore4(x2_1)
        x1_1 = torch.cat([x_up, x1], dim=1)
        x1_1 = self.relu11(self.bn11(self.conv11(x1_1)))

        # x4 = self.SPPF4(x4)
        # x_up = self.upscore2(x4)
        # x3_1 = torch.cat([x_up, x3], dim=1)
        # # x3_1 = self.dropout2d_2(x3_1)
        # # x3_1 = self.relu31(self.bn31(self.conv31(x3_1)))
        # x3_1 = self.mutisampledropout2d_2(x3_1)
        #
        # x_up = self.upscore2(x3_1)
        # x2_2 = torch.cat([x_up, x2_1, x2], dim=1)
        # x2_2 = self.relu22(self.bn22(self.conv22(x2_2)))
        #
        # x2_2_out = self.conv22_1(x2_2)
        #
        # x_up = self.upscore4(x2_2)
        # x1_2 = torch.cat([x_up, x1_1, x1], dim=1)
        # x1_2 = self.relu12(self.bn12(self.conv12(x1_2)))

        # output
        x_up = self.upscore2(x1_1)
        x0_1 = self.process1(x_up)
        # x_up = self.upscore2(x1_2)
        # x0_2 = self.process2(x_up)

        # x0_2 = torch.cat([x0_2, x0_1], dim=1)
        # x0_2 = self.conv_02_1(x0_2)
        # x0_1 = self.conv_01_1(x0_1)

        # x0_2_out = torch.cat([x0_2, x0_1], dim=1)
        # x0_2_out = self.conv_02_1(x0_2_out)
        x0_1_out = self.conv_01_1(x0_1)

        # x2_1_out = F.interpolate(x2_1_out, (x0_1_out.shape[2], x0_1_out.shape[3]), mode="bilinear", align_corners=True)
        # x2_2_out = F.interpolate(x2_2_out, (x0_2_out.shape[2], x0_2_out.shape[3]), mode="bilinear", align_corners=True)

        return x0_1_out  # , x2_1_out, x0_2_out, x2_2_out


class ShuffleNetV2(nn.Module):
    def __init__(
            self,
            stages_repeats: List[int],
            stages_out_channels: List[int],
            num_classes: int = 1000,
            mask_classes: int = 1,
            inverted_residual: Callable[..., nn.Module] = InvertedResidual,
            **kwargs
    ) -> None:
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(232 + output_channels, num_classes)
        self.fc0 = nn.Linear(232, num_classes)
        self.mutisampledropout1d0 = MultiSampleDropout(self.fc0, dim=1, p=0.2)
        self.mask_branch = MaskBranch(mask_classes=mask_classes, concat_ch=24)

        # self.conv0 = nn.Sequential(
        #     nn.Conv2d(6, 3, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(inplace=True),
        # )

    def conv_group_bn(self, inp, kernel, stride, padding=1):
        return nn.Sequential(
            nn.Conv2d(inp, inp, kernel, stride, padding, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True))

    def _forward_impl(self, x: Tensor):
        # x0 = self.conv0(x)
        # x1 = self.conv1(x0)  # (b,24,112,112)
        x1 = self.conv1(x)  # (b,24,112,112)
        x2 = self.maxpool(x1)  # (b,24,56,56)
        x2 = self.stage2(x2)  # (b,116.28,28) x 1/8
        x3 = self.stage3(x2)  # (b,232,14,14) x 1/16
        x4 = self.stage4(x3)  # (b,464,7,7) x 1/ 32
        x4 = self.conv5(x4)  # (b,1024,7,7)
        x0 = x3.mean([2, 3])
        # x = x4.mean([2, 3])# globalpool
        # x = torch.cat([x0, x], dim=1)
        # x_cls1 = self.fc0(x0)
        # x_cls2 = self.fc(x)
        x_cls1 = self.mutisampledropout1d0(x0)

        # return x_cls1, x_cls2, x1, x2, x3, x4
        return x_cls1, x1, x2, x3, x4

    def forward(self, x: Tensor):
        cls1, x1, x2, x3, x4 = self._forward_impl(x)  #
        mask1 = self.mask_branch(x1, x2, x3, x4)  #
        # return [[cls1, cls2],[mask1,mask2]]
        # cls1 = F.softmax(cls1, dim=1)
        mask1 = torch.argmax(mask1, dim=1).unsqueeze(1).float()
        return [cls1, mask1]


def _shufflenetv2(arch: str, pretrained: bool, progress: bool, *args: Any, **kwargs: Any) -> ShuffleNetV2:
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            for key in ['fc.weight', 'fc.bias']:
                state_dict.pop(key)
            model.load_state_dict(state_dict, strict=False)

    return model


def shufflenet_v2_x0_5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained, progress,
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)


if __name__ == '__main__':
    inputs = torch.randn((2, 3, 150, 300))
    model = shufflenet_v2_x1_0()
    with torch.no_grad():
        output = model(inputs)
        print(output)
