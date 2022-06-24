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

        print("size of out:{}".format(out.shape))
        print("size of residual:{}".format(residual.shape))

        out += residual

        return out


class resnet(nn.Module):
    def __init__(self, depth=56, dataset='cifar10', cfg=None):
        super(resnet, self).__init__()
        if depth == 20 or depth == 56 or depth == 110:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6
            block = BasicBlock
            if cfg is None:
                # Construct config variable.
                cfg = [[16, 16], [16, 16] * (n - 1),
                       [16, 32], [32, 32] * (n - 1),
                       [32, 64], [64, 64] * (n - 1)]
                cfg = [item for sub_list in cfg for item in sub_list]
                print('Resenet {} cfg:{}'.format(depth, cfg))
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:2 * n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[2 * n:4 * n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[4 * n:6 * n], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.select = channel_selection(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        '''
        elif depth == 50:
            block = Bottleneck
        def resnet50(pretrained=False, num_classes=10):
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
   
        elif depth == 164:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
            block = Bottleneck
        if cfg is None:
            # Construct config variable.
            cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1),
                   [128, 64, 64], [256, 64, 64]*(n-1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]
            print('now cfg:{}'.format(cfg))
        '''
        '''
        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:3*n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[6*n:9*n], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.select = channel_selection(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        '''
        if dataset == 'cifar10':
            self.num_class = 10
            #self.fc = nn.Linear(63, 10)
            self.fc = nn.Linear(64 * block.expansion, 10)
            #self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.num_class = 100
            self.fc = nn.Linear(cfg[-1], 100)

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
            )
        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:2], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[2*i: 2*(i+1)]))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = x.view(-1, 63)
        x = self.fc(x)

        return x