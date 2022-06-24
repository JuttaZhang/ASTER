import math
import torch.nn as nn
from .channel_selection import channel_selection


__all__ = ['resnet']

"""
preactivation resnet with bottleneck design.
"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        # BN-ReLU-Conv1
        #print("size of x:{}".format(x.shape))
        out = self.bn1(x)
        #print("size of bn1:{}".format(out.shape))
        out = self.select(out)
        #print("size of select:{}".format(out.shape))
        out = self.relu(out)
        #print("size of relu:{}".format(out.shape))
        out = self.conv1(out)
        #print("size of conv1:{}".format(out.shape))

        # BN-ReLU-Conv2
        out = self.bn2(out)
        #print("size of bn2:{}".format(out.shape))
        out = self.relu(out)
        #print("size of relu:{}".format(out.shape))
        out = self.conv2(out)
        #print("size of conv2:{}".format(out.shape))

        if self.downsample is not None:
            residual = self.downsample(x)

        #print("size of out:{}".format(out.shape))
        #print("size of residual:{}".format(residual.shape))
        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #print("size of out:{}".format(out.shape))
        #print("size of residual:{}".format(residual.shape))

        out += residual

        return out


class resnet(nn.Module):
    def __init__(self, depth=50, dataset='imagenet', cfg=None):
        super(resnet, self).__init__()
        if depth == 50:
            layers = [3, 4, 6, 3]
            block = Bottleneck
            if cfg is None:
                cfg = [[64, 64, 64],[256, 64, 64]*(layers[0]-1),
                      [256, 128, 128],[512, 128, 128]*(layers[1]-1),
                      [512, 256, 256],[1024, 256, 256]*(layers[2]-1),
                      [1024, 512, 512], [2048, 512, 512]*(layers[3]-1)]
                cfg = [item for sub_list in cfg for item in sub_list]
                print('Resenet {} cfg:{}'.format(depth, cfg))
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False)

        #self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:2 * n])
        #self.layer2 = self._make_layer(block, 32, n, cfg=cfg[2 * n:4 * n], stride=2)
        #self.layer3 = self._make_layer(block, 64, n, cfg=cfg[4 * n:6 * n], stride=2)

        self.layer1 = self._make_layer(block, 64, layers[0], cfg=cfg[0: 3 * layers[0]])
        self.layer2 = self._make_layer(block, 128, layers[1], cfg=cfg[3 * layers[0]:3*sum(layers[0:2])]
                                       , stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg=cfg[3*sum(layers[0:2]):3*sum(layers[0:3])],
                                       stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg=cfg[3*sum(layers[0:3]):3*sum(layers[0:4])],
                                       stride=2)


        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.select = channel_selection(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        #self.avgpool = nn.AvgPool2d(8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if dataset == 'imagenet':
            self.num_class = 1000
            self.fc = nn.Linear(512 * block.expansion, self.num_class)
        else:
            raise ValueError("Only support Imagenet!")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = x.view(-1, 45)
        x = self.fc(x)

        return x