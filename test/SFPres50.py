import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

#用于resnet50模型的软剪枝方法

import numpy as np

import sys

from channel_selection import *
from resnet_imagenet import *
model_names = ""

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--thres', default=0.1, type=float, help='threshold to prune')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def softpruning(args, model, thres, end_flag):
    total = 0  # out_channel数量
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    bn = torch.zeros(total)  # 把所以bn层的｜weight｜合到同一个tensor中
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size
    y, i = torch.sort(bn)  # y表示排序后的原tensor，i表示排在第几位
    y = y.cuda()
    thre_index = 0
    for i in range(len(y)):
        if abs(y[i]) < thres:
            thre_index += 1
    ratio = thre_index / total
    thre = y[thre_index]  # 低于thre的全部裁剪掉

    pruned = 0

    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.clone()
            mask = weight_copy.abs().gt(thre).float().cuda()
            # torch.sum(mask)统计大于thre的数，mask.shape[0] - torch.sum(mask)为小于thre需要被裁剪掉的数目
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            # 对于需要被裁剪掉的channel,w,b直接置零
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            m.running_mean.mul_(mask)
            m.running_var.mul_(mask)
            cfg_mask.append(mask.clone())
            if (int(torch.sum(mask))/mask.shape[0]<=0.1):
                end_flag = True
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
    conv_mask = []
    linear_mask = []
    old_modules = list(model.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            # idx1表示mask=1的index
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):

            if isinstance(old_modules[layer_id - 1], channel_selection) or isinstance(old_modules[layer_id - 1],
                                                                                      nn.BatchNorm2d):
                print(layer_id, m0, conv_count)
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                #print('shape: {} shape:{}'.format(idx0.shape, idx1.shape))
                if idx0.shape == ():
                    inshape = 1
                else:
                    inshape = idx0.shape[0]
                if idx1.shape == ():
                    outshape = 1
                else:
                    outshape = idx1.shape[0]
                print('In shape: {:d} Out shape:{:d}'.format(inshape, outshape))

                for i in range(m0.weight.data.shape[1]):
                    if i not in idx0:
                        m0.weight.data[:, i, :, :] = 0
                    if conv_count % 3 != 1:
                        for j in range(m0.weight.data.shape[0]):
                            if j not in idx1 and i in idx0:
                                m0.weight.data[j, i, :, :] = 0
                mask = m0.weight.data.ne(0).float().cuda()
                conv_mask.append(mask.clone())
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            for i in range(m0.weight.data.shape[1]):
                if i not in idx0:
                    m0.weight.data[:, i] = 0
            mask = m0.weight.data.ne(0).float().cuda()
            linear_mask.append(mask.clone())

    return model, ratio,cfg_mask,conv_mask,linear_mask,end_flag

def do_Mask(model,cfg_mask,conv_mask,linear_mask):
    i = 0
    j = 0
    l = 0
    conv_count = 0
    old_modules = list(model.modules())
    for layer_id in range(len(old_modules)):
        m = old_modules[layer_id]
        if isinstance(m, nn.BatchNorm2d) and i < len(cfg_mask):
            print('m.weight.data shape:{}, bn_mask shape:{} '.format(m.weight.data.shape, cfg_mask[i].shape))
            m.weight.data.mul_(cfg_mask[i])
            m.bias.data.mul_(cfg_mask[i])
            m.running_mean.mul_(cfg_mask[i])
            m.running_var.mul_(cfg_mask[i])
            i+=1
        elif isinstance(m, nn.Conv2d) and j < len(conv_mask):
            if conv_count == 0:
                conv_count += 1
                continue
            elif  isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
                print('m.weight.data shape:{}, conv_mask shape:{} '.format(m.weight.data.shape, conv_mask[j].shape))
                m.weight.data.mul_(conv_mask[j])
                if m.bias is not None:
                    m.bias.data.mul_(conv_mask[j])
                j+=1
        elif isinstance(m, nn.Linear) and l < len(linear_mask):
            print('m.weight.data shape:{}, linear_mask shape:{} '.format(m.weight.data.shape, linear_mask[l].shape))
            m.weight.data.mul_(linear_mask[l])
            if l != len(linear_mask)-1:
                m.bias.data.mul_(linear_mask[l])
            l+=1
    return model

if __name__ == '__main__':
    args = parser.parse_args()
    ratio=0.
    cfg_mask = []
    conv_mask = []
    linear_mask = []
    end_flag = True
    model = resnet(depth=50, dataset='imagenet').cuda()
    print(model)
    #model, ratio, cfg_mask, conv_mask, linear_mask, end_flag = softpruning(args, model, 0.4, end_flag)
