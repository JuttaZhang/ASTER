import numpy as np
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import sys
import os
sys.path.append('../')
from models.vggs import vgg
from models.resnetex import *
from tensorboardX import SummaryWriter

def showtensorboard(model, tb_writer):
    for name, param in model.named_modules():
        if isinstance(param, nn.BatchNorm2d):
            print(param.weight.data.abs())
            #tb_writer.add_histogram(os.path.join('Resnet56_',name), param.weight.data.abs().clone(), global_step=1)
            tb_writer.add_histogram(os.path.join('VGG16_', name), param.weight.data.abs().clone(), global_step=1)

#checkpoint1 = torch.load('../baselines/baseline_resnet56.pth.tar')
#model1 = resnet(depth=56,dataset='cifar10')
#model1.load_state_dict(checkpoint1['state_dict'])

checkpoint2 = torch.load('../baselines/baseline_vgg16_cifar100.pth.tar')
model2 = vgg(depth=16,dataset='cifar100')
model2.load_state_dict(checkpoint2['state_dict'])
# 画出每个卷积层的分布
tb_writer = SummaryWriter('')
#showtensorboard(model1,tb_writer)
showtensorboard(model2,tb_writer)



