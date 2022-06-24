import os
import math
import torch
import torch.nn as nn
from typing import List
import torchvision.models
__all__ = ['ResNet', 'resnet50']

# you need to download the models to ~/.torch/models
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
models_dir = os.path.expanduser('~/.torch/models')
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg: List[int], stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, cfg=None, num_classes=1000):
        super(ResNet, self).__init__()
        if cfg is None:
            cfg = [[64, 64, 64], [256, 64, 64] * (layers[0] - 1),
                   [256, 128, 128], [512, 128, 128] * (layers[1] - 1),
                   [512, 256, 256], [1024, 256, 256] * (layers[2] - 1),
                   [1024, 512, 512], [2048, 512, 512] * (layers[3] - 1)]
            cfg = [item for sub_list in cfg for item in sub_list]
        print(cfg)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cfg=cfg[0: 3 * layers[0]])
        self.layer2 = self._make_layer(block, 128, layers[1], cfg=cfg[3 * layers[0]:3*sum(layers[0:2])], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg=cfg[3*sum(layers[0:2]):3*sum(layers[0:3])], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg=cfg[3*sum(layers[0:3]):3*sum(layers[0:4])], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3 * i: 3 * (i + 1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet50(pretrained=False, cfg=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], cfg=cfg, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet50'])))
    return model
