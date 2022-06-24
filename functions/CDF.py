import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import sys
sys.path.append('../')
from models.vggs import vgg
from models.resnetex import *


def count_total(model):
    total = 0  # out_channel数量
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    return total

def com_ratio(model,thres):
    total = count_total(model)
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size
    bn = torch.abs(bn)
    y, i = torch.sort(bn)  # y表示排序后的原tensor，i表示排在第几位
    thre_index = 0
    for i in range(len(y)):
        if abs(y[i]) < thres:
            thre_index += 1
    ratio = thre_index / total
    '''for i in range(len(y)):
        if abs(y[i]) != 0:
            thres = y[i]
            break'''
    return ratio
def draw_CDF(model):
    y = [com_ratio(model, n) for n in np.arange(0,1.01,0.01)]
    x = np.arange(0,1.01,0.01)
    plt.plot(x,y)
    plt.xlabel('Threshold ')
    plt.ylabel('Prune rate')
    plt.title('Resnet110')
    plt.show()
    return

#checkpoint = torch.load('../baselines/baseline_resnet56.pth.tar')
#model = vgg(depth=16, dataset='cifar100', cfg=checkpoint['cfg'])
#model = resnet(depth=110, cfg=checkpoint['cfg'])
#model = resnet(depth=56,cfg=checkpoint['cfg'])
#model.load_state_dict(checkpoint['state_dict'])


checkpoint = torch.load('../baselines/baseline_resnet110.pth.tar')
#model = vgg(depth=16, dataset='cifar100')
model = resnet(depth=110,dataset='cifar10')
model.load_state_dict(checkpoint['state_dict'])
draw_CDF(model)

