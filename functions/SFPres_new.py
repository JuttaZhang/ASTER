import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

#用于resnet模型的软剪枝方法

import numpy as np

import sys
sys.path.append('../')
from models.channel_selection import *
from models.resnetex import *

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model, args):
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    # print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
    #    correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def softpruning(args, model, thres, end_flag):
    total = 0  # out_channel数量
    y = []
    index = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
            y_temp,i_temp = torch.sort(m.weight.data.abs().clone())
            #print('before{}'.format(m.weight.data.abs()))
            #print('sort{}'.format(y_temp))
            #print('index{}'.format(i_temp))
            y.append(y_temp)
            index.append(i_temp)

    thre_index = [0]*55
    pruned = 0
    #print('len sorted y:{}'.format(len(y)))
    for i in range(len(y)):
        for j in range(y[i].shape[0]):
            if abs(y[i][j]) < thres[i]:
                thre_index[i] += 1
            #print(thre_index[i])
        pruned += thre_index[i]
    #print('has been pruned:{}'.format(pruned))

    ratio = pruned / total
    #thre = y[thre_index]  # 低于thre的全部裁剪掉
    thre = []
    for i in range(len(y)):
        if thre_index[i] == 0:
            thre.append(thres[i])
        else:
            thre.append(y[i][thre_index[i]-1])
    print("threshold:{}".format(thre))

    cfg_mask = []
    i = 0
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.clone()
            mask = weight_copy.abs().gt(thre[i]).float().cuda()
            # torch.sum(mask)统计大于thre的数，mask.shape[0] - torch.sum(mask)为小于thre需要被裁剪掉的数目
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            # 对于需要被裁剪掉的channel,w,b直接置零
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            m.running_mean.mul_(mask)
            m.running_var.mul_(mask)
            cfg_mask.append(mask.clone())
            if (int(torch.sum(mask))/mask.shape[0]<=0.0625):
                end_flag = True
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
            i+=1
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
                    if conv_count % 2 != 1:
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

    after_prune_acc = test(model, args)
    #print('after_prune_acc:{}'.format(after_prune_acc))

    return model, ratio, after_prune_acc,cfg_mask,conv_mask,linear_mask,end_flag

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

