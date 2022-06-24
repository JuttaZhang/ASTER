from __future__ import print_function
import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorboardX import SummaryWriter
from tensorboard_logger import Logger
import shutil
import copy
import math

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import evaluate

import sys
sys.path.append('../')

from models.vggs import vgg
from models.resnetex import *


from functions.reLBFGS import LBFGS

from functions.utils import lbfgs_alpha_showtensorboard,gensubgradient,compute_stats,get_grad,prune_ratio,newthres,find_upper_bound
#------------------------------------------
#使用传统方法的补充实验，该实验主要关注精确度的变化
#使用的判断实验停止的方法是，如果某一层中剩余的channel少于1%就停止剪枝
#结果存放位置 /results
#------------------------------------------
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--model', type=str, default='resnet18',
                    help='model (default: resnet18)')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--num', type=int, default=1,
                    help='no')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='refine from prune model')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lb', type=float, default=0.8,
                    help='lower bound of pruning ratio (default: 0)')
parser.add_argument('--ub', type=float, default=1,
                    help='upper bound of pruning ratio (default: 0)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')


def updateBN(s):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(s*torch.sign(m.weight.data))  # L1

#is_bn用来判断当前需要获取的参数是指bn层的参数还是其余层的参数
def get_bnparameters(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            yield m.weight


def test(model,test_loader,f):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),file=f)
    return correct / float(len(test_loader.dataset))

#保存预测精度最高的稀疏模型
def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:#model_best
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


def train_evaluate(parameterization):
    args = parser.parse_args()
    torch.cuda.manual_seed(1)
    # 加载数据
    kwargs = {'num_workers': 2, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = vgg(depth=16, dataset='cifar10')
    model.cuda()
    args.s = parameterization.get("lambda", 0.0001)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    filepath = './results/{}'.format(args.s)
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    f = open(os.path.join(filepath, 'event.log'), 'wt')
    logger = Logger(logdir=filepath, flush_secs=10)
    best_tradeoff = 0.
    ratio = 0
    best_prec = 0.
    filepath_s= filepath +'_sparse'
    if not os.path.exists(filepath_s):
        os.mkdir(filepath_s)
    for epoch in range(0, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            if args.sr:
                updateBN(args.s)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()), file=f)
        dtype = torch.float
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        prec1 = evaluate(
            net=model,
            data_loader=test_loader,
            dtype=dtype,
            device=device,
        )
        is_best1 = prec1 > best_prec
        best_prec = max(prec1, best_prec)
        if is_best1:
            best_prec = prec1
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best1, filepath=filepath_s)

    #model_new
    model = vgg(depth=16, dataset='cifar10')
    model.cuda()
    args.model = os.path.join(filepath_s, 'model_best.pth.tar')
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model),file=f)
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                      .format(args.model, checkpoint['epoch'], best_prec1),file=f)
    print(model,file=f)
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    bn = torch.zeros(total)
    index = 0
    thre = 10000
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            torch.sort(bn)
            size = m.weight.data.shape[0]
            val, _ = m.weight.data.topk(k=int(0.01*size)+1, largest=True, sorted=True)
            #print(val)
            y,_ = torch.sort(val, descending=False)
            min = y[0]
            #print(min)
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size
            if min < thre:
                thre = min
            #print(thre)
    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))),file=f)
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    ratio = pruned / total
    print('Pre-processing Successful!',file=f)

    newmodel = vgg(dataset=args.dataset, cfg=cfg)
    newmodel.cuda()

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size), file=f)
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
    epoch = 0
    prec2 = 0.
    best_prec1 = 0.
    newoptimizer = optim.SGD(newmodel.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(0, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
            for param_group in newoptimizer.param_groups:
                param_group['lr'] *= 0.1
        newmodel.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            newoptimizer.zero_grad()
            output = newmodel(data)
            newloss = F.cross_entropy(output, target)
            newloss.backward()
            newoptimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), newloss.item()), file=f)
        dtype = torch.float
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        prec2 = evaluate(
            net=newmodel,
            data_loader=test_loader,
            dtype=dtype,
            device=device,
        )
        tradeoff = prec2
        is_best = tradeoff >= best_tradeoff
        if is_best:
            best_prec1 = prec2
            best_pratio = ratio
        best_tradeoff = max(tradeoff, best_tradeoff)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': prec2,
            'best_pratio': ratio,
            'best_tradeoff': tradeoff,
            'optimizer': optimizer.state_dict(),
        }, is_best,filepath=filepath)
    print("The best tradeoff performance of this round, best_prec1: {}, best_pratio: {}".format(best_prec1, best_pratio))
    return best_tradeoff



best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lambda", "type": "range", "bounds": [1e-8, 0.1], "log_scale": True},
        #{"name": "pruning_ratio", "type": "range", "bounds": [0, 1], "log_scale": True}
    ],
    evaluation_function=train_evaluate,
    objective_name='lambda',
    total_trials=10,
)

print(best_parameters)
means, covariances = values
print(means)
print(covariances)

best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])

#best_objective_plot = optimization_trace_single_method(
#    y=np.maximum.accumulate(best_objectives, axis=1),
#    title="Model performance vs. # of iterations",
#    ylabel="Tradeoff(prec+ratio), %",
#)
#render(best_objective_plot)
#render(plot_contour(model=model, param_x='init_thres', param_y='x', metric_name='tradeoff'))

data = experiment.fetch_data()
df = data.df
best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
best_arm = experiment.arms_by_name[best_arm_name]
print(best_arm)

train_evaluate(best_arm.parameters)