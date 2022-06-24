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

import sys
sys.path.append('../')

from models.vggs import vgg
from models.resnetex import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--model', type=str, default='resnet18',
                    help='model (default: resnet18)')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')

parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='refine from prune model')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#if not os.path.exists(args.save):
#    os.makedirs(args.save)

#加载数据
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
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
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', download=True, train=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                          ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', download=True, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

num_classes = 10 if args.dataset == 'cifar10' else 100

#model = torch.load("modelsparseSGD", map_location=lambda storage, loc: storage)

#Fine-tune
if args.refine:
    checkpoint = torch.load(args.refine)
    if args.model == 'vgg11':
        model = vgg(depth=11, dataset=args.dataset)
    elif args.model == 'vgg16':
        model = vgg(depth=16, dataset=args.dataset)
    elif args.model == 'resnet56':
        model = resnet(depth=56, dataset=args.dataset)
    elif args.model == 'resnet110':
        model = resnet(depth=110, dataset=args.dataset)
    #model = vgg(cfg=checkpoint['cfg'])
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
else:
    if args.model == 'vgg11':
        model = vgg(depth=11, dataset=args.dataset)
    elif args.model == 'vgg16':
        model = vgg(depth=16, dataset=args.dataset)
    elif args.model == 'resnet56':
        model = resnet(depth=56, dataset=args.dataset)
    elif args.model == 'resnet110':
        model = resnet(depth=110, dataset=args.dataset)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


if args.resume:
    if os.path.isfile(args.resume):#判断某一对象(需提供绝对路径)是否为文件
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)#从文件中加载一个用torch.save()保存的对象
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


def train(model,epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return model

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        output = model(data)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

#保存预测精度最高的稀疏模型
def save_checkpoint(state, is_best, filename='baseline_{}_{}/checkpoint.pth.tar'.format(args.model,args.dataset)):
    torch.save(state, filename)
    if is_best:#model_best
        shutil.copyfile(filename, 'baseline_{}_{}.pth.tar'.format(args.model,args.dataset))

logger = Logger(logdir="./baseline_{}_{}".format(args.model,args.dataset), flush_secs=10)
best_prec1 = 0.
t0 = time.time()
tb_writer = SummaryWriter()
ratio=0

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    model = train(model,epoch)
    if epoch == args.epochs-1:
        print("The sparse took", time.time() - t0)
    prec1 = test()
    logger.log_value('test_acc', prec1, epoch)
    is_best = prec1 >= best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best)