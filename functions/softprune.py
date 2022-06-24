import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models.vgg import vgg
import numpy as np
import copy

#最初的预剪枝方法，不对卷积层以及全连接层进行处理，仅针对BN进行置零操作
# soft-Prune settings
'''
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
#设置裁剪率
parser.add_argument('--percent', type=float, default=0.5,
                    help='prune ratio (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to raw trained model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
'''
'''
model = vgg()
if args.cuda:
    model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print(model)
'''
def prox_l1_norm(w, lamb):
    """perform soft-thresholding on input vector"""
    if type(lamb)!= float:
        temp = np.sign(w) * np.maximum(np.abs(w) - lamb.detach().cpu().numpy(), 0)
    else:
        temp = np.sign(w)*np.maximum(np.abs(w)-lamb,0)
    return torch.from_numpy(temp).cuda()

def softpruning(args,model,thres):
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
    cfg = []  # 每层中保留下来的channel，'M'表示maxpool层
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
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
            #z = copy.deepcopy(m.weight.data.cpu()).numpy()
            #m.weight.data = prox_l1_norm(z, thres)
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    after_prune_acc = test(model,args)
    return model,ratio,after_prune_acc, cfg_mask

def do_Mask(model,cfg_mask):
    i = 0
    old_modules = list(model.modules())
    for layer_id in range(len(old_modules)):
        m = old_modules[layer_id]
        if isinstance(m, nn.BatchNorm2d) and i < len(cfg_mask):
            print('m.weight.data shape:{}, bn_mask shape:{} '.format(m.weight.data.shape, cfg_mask[i].shape))
            m.weight.data.mul_(cfg_mask[i])
            m.bias.data.mul_(cfg_mask[i])
            m.running_mean.mul_(cfg_mask[i])
            m.running_var.mul_(cfg_mask[i])
            i += 1
    return model

'''

total = 0 #out_channel数量
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]


bn = torch.zeros(total)#把所以bn层的｜weight｜合到同一个tensor中
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn) #y表示排序后的原tensor，i表示排在第几位
thre_index=0
for i in range(len(y)):
    if abs(y[i]) < 0.0001:
        thre_index += 1
print(total)
print('*'*50)
print(thre_index)
print('prune_ratio:{}%'.format((thre_index/total)*100))
#thre_index = int(total * args.percent)
thre = y[thre_index]#低于thre的全部裁剪掉

pruned = 0
cfg = []#每层中保留下来的channel，'M'表示maxpool层
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
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

print('Pre-processing Successful!')

'''

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model,args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    #print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
    #    correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))
'''
test()

# Make prune mask
print(cfg)

conv_mask=[]
linnear_mask=[]

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for m0 in model.modules():
    if isinstance(m0, nn.BatchNorm2d):
        #idx1表示mask=1的index
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
        for i in range(m0.weight.data.shape[1]):
            if i not in idx0:
                m0.weight.data[:,i,:,:] = 0
            for j in range(m0.weight.data.shape[0]):
                if j not in idx1 and i in idx0:
                    m0.weight.data[j, i, :, :] = 0
        mask = m0.weight.data.ne(0).float().cuda()
        #print(mask.shape)
        #print(int(torch.sum(mask)))
        conv_mask.append(mask.clone())
        # m1.bias.data = m0.bias.data[idx1].clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        for i in range(m0.weight.data.shape[1]):
            if i not in idx0:
                m0.weight.data[:, i] = 0
        mask = m0.weight.data.ne(0).float().cuda()
        #m0.bias.data.mul_(mask)
        #print(int(torch.sum(mask)))
        linnear_mask.append(mask.clone())

print(model)

torch.save({'cfg': cfg, 'state_dict': model.state_dict()}, args.save)

test()
torch.save(model, "modelpruned")

'''
