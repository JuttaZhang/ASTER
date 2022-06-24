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
    kwargs = {'num_workers': 2, 'pin_memory': True}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', download=True, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    # print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
    #    correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def softpruning(f,args, model, thres, end_flag, output_size):
    sub_flops = []
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
            if (int(torch.sum(mask))/mask.shape[0]<=0.05): #renet56 58
            #if (int(torch.sum(mask)) / mask.shape[0] <= 0.1):  #renet110
                end_flag = True
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))),file=f)
    conv_mask = []
    linear_mask = []
    old_modules = list(model.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0
    conv_id = 0

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
            output_height, output_width = output_size[conv_id]
            if isinstance(old_modules[layer_id - 1], channel_selection) or isinstance(old_modules[layer_id - 1],
                                                                                      nn.BatchNorm2d):
                output_height, output_width = output_size[conv_id]
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
                #print('In shape: {:d} Out shape:{:d}'.format(inshape, outshape),file=f)

                for i in range(m0.weight.data.shape[1]):
                    if i not in idx0:
                        m0.weight.data[:, i, :, :] = 0
                    if conv_count % 2 != 1:
                        for j in range(m0.weight.data.shape[0]):
                            if j not in idx1 and i in idx0:
                                m0.weight.data[j, i, :, :] = 0
                mask = m0.weight.data.ne(0).float().cuda()
                conv_mask.append(mask.clone())
                kernel_ops = m0.kernel_size[0] * m0.kernel_size[1] * (inshape / m0.groups)
                flops = kernel_ops * outshape * output_height * output_width
            else:
                kernel_ops = m0.kernel_size[0] * m0.kernel_size[1] * (m0.in_channels / m0.groups)
                flops = kernel_ops * m0.out_channels * output_height * output_width
            conv_id += 1
            sub_flops.append(flops)
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            for i in range(m0.weight.data.shape[1]):
                if i not in idx0:
                    m0.weight.data[:, i] = 0
            mask = m0.weight.data.ne(0).float().cuda()
            linear_mask.append(mask.clone())
            flops = idx0.shape[0] * m0.out_features
            sub_flops.append(flops)
    total_flops = sum(sub_flops)
    print("total_flops========>{}".format(total_flops),file=f)
    after_prune_acc = test(model, args)

    return model, total_flops, ratio, after_prune_acc,cfg_mask,conv_mask,linear_mask,end_flag

def do_Mask(f,model,cfg_mask,conv_mask,linear_mask):
    i = 0
    j = 0
    l = 0
    conv_count = 0
    old_modules = list(model.modules())
    for layer_id in range(len(old_modules)):
        m = old_modules[layer_id]
        if isinstance(m, nn.BatchNorm2d) and i < len(cfg_mask):
            print('m.weight.data shape:{}, bn_mask shape:{} '.format(m.weight.data.shape, cfg_mask[i].shape),file=f)
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
                #print('m.weight.data shape:{}, conv_mask shape:{} '.format(m.weight.data.shape, conv_mask[j].shape),file=f)
                m.weight.data.mul_(conv_mask[j])
                if m.bias is not None:
                    m.bias.data.mul_(conv_mask[j])
                j+=1
        elif isinstance(m, nn.Linear) and l < len(linear_mask):
            #print('m.weight.data shape:{}, linear_mask shape:{} '.format(m.weight.data.shape, linear_mask[l].shape),file=f)
            m.weight.data.mul_(linear_mask[l])
            if l != len(linear_mask)-1:
                m.bias.data.mul_(linear_mask[l])
            l+=1
    return model

