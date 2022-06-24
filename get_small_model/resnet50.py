from __future__ import absolute_import

import math
import numpy as np
import torch
import torch.nn as nn
import typing
from typing import List

__all__ = ['resnet50']



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, outplanes: int, cfg: List[int], stride=1, downsample=None,
                 expand=False):
        """

        :param inplanes: the input dimension of the block
        :param outplanes: the output dimension of the block
        :param cfg: the output dimension of each convolution layer
            config format:
            [conv1_out, conv2_out, conv3_out, conv1_in]
        :param gate: if use gate between conv layers
        :param stride: the stride of the first convolution layer
        :param downsample: if use the downsample convolution in the residual connection
        :param expand: if use ChannelExpand layer in the block
        """
        super(Bottleneck, self).__init__()

        conv_in = cfg[3]
            # the main body of the block

            # add a SparseGate before the first conv
            # to enable the pruning of the input dimension for further reducing computational complexity

        self.conv1 = nn.Conv2d(conv_in, cfg[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])

        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])

        self.conv3 = nn.Conv2d(cfg[1], cfg[2], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        #self.expand = expand
        #self.expand_layer = ChannelExpand(outplanes) if expand else None

        #self.tanh = nn.Tanh()


    def forward(self, x):
        residual = x

        out = self.select(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.downsample(x)
        #if self.expand_layer:
        #    out = self.expand_layer(out)

        out += residual
        out = self.relu(out)
        #out = self.tanh(out)

        return out

class ResNetExpand(nn.Module):
    def __init__(self, layers=None, cfg=None, downsample_cfg=None, mask=False,
                 width_multiplier=1.0, expand_idx=None):
        super(ResNetExpand, self).__init__()

        if mask:
            raise NotImplementedError("do not support mask layer")

        if layers is None:
            # resnet 50
            layers = [3, 4, 6, 3]
        if len(layers) != 4:
            raise ValueError("resnet should be 4 blocks")

        # if width_multiplier != 1.0 and cfg is not None:
        #     raise ValueError("custom cfg is conflict with width_multiplier")
        self.width_multiplier = width_multiplier

        self.flops_weight_computed = False  # if the conv_flops_weight has been computed


        block = Bottleneck

        # config format:
        # the config of each bottleneck is a list with length = 4
        # [conv1_out, conv2_out, conv3_out, conv1_in]
        self._cfg_len = 4
        default_cfg = self._default_config(layers)
        for i in range(len(default_cfg)):
            # multiply the width_multiplier
            for j in range(len(default_cfg[i])):
                default_cfg[i][j] = int(default_cfg[i][j] * width_multiplier)

            if len(default_cfg[i]) != 1:  # skip the first layer (model.conv1)
                # check the config length
                if len(default_cfg[i]) != self._cfg_len:
                    raise ValueError(f"Each block should have {self._cfg_len} layer, got {len(default_cfg[i])}")

        # flatten the config
        default_cfg = [item for sub_list in default_cfg for item in sub_list]

        default_downsample_cfg = [256, 512, 1024, 2048]
        if not downsample_cfg:
            downsample_cfg = default_downsample_cfg
        for i in range(len(downsample_cfg)):
            downsample_cfg[i] = int(downsample_cfg[i] * width_multiplier)
        assert len(downsample_cfg) == len(default_downsample_cfg)

        if cfg is None:
            # Construct config variable.
            cfg = default_cfg
        assert len(cfg) == len(default_cfg), f"Config length error! Expected {len(default_cfg)}, got {len(cfg)}"

        # dimension of residuals
        # self.planes = [256, 512, 1024, 2048]
        # assert len(self.planes) == 4
        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=3, stride=1, padding=1,
                              bias=False)
        #self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block,
                                       inplanes=cfg[0],
                                       outplanes=int(256 * width_multiplier),
                                       blocks=layers[0],
                                       downsample_cfg=downsample_cfg[0],
                                       cfg=cfg[1:self._cfg_len * layers[0] + 1],
                                       downsample_out=int(256 * width_multiplier))
        self.layer2 = self._make_layer(block,
                                       inplanes=int(256 * width_multiplier),
                                       outplanes=int(512 * width_multiplier),
                                       blocks=layers[1],
                                       downsample_cfg=downsample_cfg[1],
                                       cfg=cfg[self._cfg_len * layers[0] + 1:self._cfg_len * sum(layers[0:2]) + 1],
                                       stride=2,
                                       downsample_out=int(512 * width_multiplier))
        self.layer3 = self._make_layer(block,
                                       inplanes=int(512 * width_multiplier),
                                       outplanes=int(1024 * width_multiplier),
                                       blocks=layers[2],
                                       downsample_cfg=downsample_cfg[2],
                                       cfg=cfg[
                                           self._cfg_len * sum(layers[0:2]) + 1:self._cfg_len * sum(layers[0:3]) + 1],
                                       stride=2,
                                       downsample_out=int(1024 * width_multiplier))
        self.layer4 = self._make_layer(block,
                                       inplanes=int(1024 * width_multiplier),
                                       outplanes=int(2048 * width_multiplier),
                                       blocks=layers[3],
                                       downsample_cfg=downsample_cfg[3],
                                       cfg=cfg[
                                           self._cfg_len * sum(layers[0:3]) + 1:self._cfg_len * sum(layers[0:4]) + 1],
                                       stride=2,
                                       downsample_out=int(2048 * width_multiplier))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(int(2048 * width_multiplier), 1000)


        '''if expand_idx:
            # set channel expand index
            if expand_idx is not None:
                for m_name, sub_module in self.named_modules():
                    if isinstance(sub_module, models.common.ChannelOperation):
                        sub_module.idx = expand_idx[m_name]'''

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()



    def _make_layer(self, block, inplanes, outplanes, blocks, cfg, downsample_cfg, downsample_out, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, downsample_cfg,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(downsample_cfg),
            #ChannelExpand(downsample_out),
        )

        layers = []
        layers.append(block(inplanes=inplanes, outplanes=outplanes,
                            cfg=cfg[:self._cfg_len], stride=stride, downsample=downsample,
                            expand=True))
        for i in range(1, blocks):
            layers.append(block(inplanes=downsample_out, outplanes=outplanes,
                                cfg=cfg[self._cfg_len * i:self._cfg_len * (i + 1)], expand=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    def config(self, flatten=True):
        cfg = [[self.conv1.out_channels]]
        for submodule in self.modules():
            if isinstance(submodule, Bottleneck):
                cfg.append(submodule.config())

        if flatten:
            flatten_cfg = []
            for sublist in cfg:
                for item in sublist:
                    flatten_cfg.append(item)

            return flatten_cfg

        return cfg

    '''def expand_idx(self) -> typing.Dict[str, np.ndarray]:
        """get the idx dict of ChannelExpand layers"""
        expand_idx: typing.Dict[str, np.ndarray] = {}
        for m_name, sub_module in self.named_modules():
            if isinstance(sub_module, models.common.ChannelOperation):
                expand_idx[m_name] = sub_module.idx

        return expand_idx'''
    def _default_config(self, layers: List[int]) -> List[List[int]]:
        # the output dimension of the conv1 in the network (NOT the block)
        conv1_output = 64

        default_cfg = [[64, 64, 256] for _ in range(layers[0])] + \
                      [[128, 128, 512] for _ in range(layers[1])] + \
                      [[256, 256, 1024] for _ in range(layers[2])] + \
                      [[512, 512, 2048] for _ in range(layers[3])]

        input_dim = conv1_output
        for block_cfg in default_cfg:
            # the conv_in is as same as the input dim by default (ChannelSelection selects all channels)
            block_cfg.append(input_dim)
            input_dim = block_cfg[2]

        default_cfg = [[conv1_output]] + default_cfg

        return default_cfg

def resnet50(width_multiplier,cfg=None):
    model = ResNetExpand(cfg=cfg, width_multiplier=width_multiplier,
                         layers=[3, 4, 6, 3])
    return model

def _test():
    """unit test of ResNet-50 model"""

    from torchvision.models import resnet50 as torch_resnet50

    print("########## Unit test of the ResNet-50 with ChannelExpand ##########")

    # test default config
    print("Testing default config")
    model = resnet50(width_multiplier=1.0)
    print(model)

    print("Test passed!")


if __name__ == '__main__':
    _test()