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
from functions.SFP_flops import softpruning,do_Mask
import copy
import math
import csv
import pandas as pd

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import evaluate
from tools import compute_conv_flops, return_output

import sys
#sys.path.append('functions/')

from models.vggs import vgg
from models.resnetex import *


from functions.reLBFGS import LBFGS

from functions.utils import lbfgs_alpha_showtensorboard,gensubgradient,compute_stats,get_grad,prune_ratio,newthres,find_upper_bound, get_bn_layer
# 尝试看\lambada的初始化是否能够实现

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--model', type=str, default='resnet18',
                    help='model (default: resnet18)')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--num', type=int, default=1,
                    help='no')
parser.add_argument('--exp_flops', type=float, default=0.4,
                    help='expert flops drop (default: 0)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='refine from prune model')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=320, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--pec', type=float, default=0.5,
                    help='衡量tradeoff的比例')
parser.add_argument('--lb', type=float, default=0.7,
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
        datasets.CIFAR10('./data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = vgg(depth=16, dataset='cifar10')
    flops_init = compute_conv_flops(model)
    print("Flops init =========>{}".format(flops_init))
    model.cuda()
    thres = parameterization.get("init_thres", 0.0001)
    init_thres = thres
    x = parameterization.get("x", 0.01)
    flops = flops_init
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    filepath = './resultsVgg16/{}_{}_{}_{}'.format(init_thres, x, args.num, args.exp_flops)
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    f = open(os.path.join(filepath, 'event.log'), 'wt')
    logger = Logger(logdir=filepath, flush_secs=10)
    best_tradeoff = 0.
    flag = True
    upper_flag = False if args.ub == 0 else True  # 硬性标准
    t0 = time.time()
    tb_writer = SummaryWriter()
    ratio = 0
    prec1 = 0
    alpha = 1
    sum_delta = 0
    thres_before = 0
    stop_epoch = 0
    pre_model = copy.deepcopy(model)
    chosen_flag = False
    cfg_mask = []
    conv_mask = []
    linear_mask = []
    end_flag = False
    for epoch in range(0, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        if epoch == args.epochs - 1:
            flag = False

        model.train()
        optimizermu = LBFGS(get_bnparameters(model), lr=args.lr, history_size=10, mu=args.s, batch_size=args.batch_size,
                            line_search_fn="strong_wolfe")

        if (epoch + 1) % 5 == 0 and flag and upper_flag and not chosen_flag:

            sum_delta = 0
            ratio = prune_ratio(model)
            print('Before update:  the prune ratio:{}  threshold:{}'.format(ratio, thres), file=f)
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                def closure():
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    return loss
                #print('===========step:{}'.format(batch_idx), file=f)
                loss, update, _ = optimizermu.step(closure)
                optimizer.step()
                sub = gensubgradient(model)
                views = []
                for subs in sub:
                    views.extend(subs)
                views = torch.Tensor(views).cuda()
                update = torch.mul(update, views)
                temp = torch.ones(update.numel()).cuda()
                derivative_l = update.dot(temp)
                delta = x * loss / derivative_l  # 0.01可以进行更改
                mix = optimizermu.param_groups[0]['lr'] * alpha * delta / args.batch_size
                #print('mix:{}'.format(mix), file=f)
                thres_before = thres
                thres = thres + mix
                sum_delta_before = sum_delta
                sum_delta += mix


                if thres > newthres(model, args.lb):
                    thres = newthres(model, args.lb)

                upper_bound = find_upper_bound(model)
                if thres >= upper_bound or thres < 0:
                    thres = thres_before  # 回滚
                    sum_delta = sum_delta_before

                print('After update:  threshold:{}'.format(thres), file=f)

                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()), file=f)

            print('sum_delta of epoch {}: {}'.format(epoch, sum_delta), file=f)
            #if sum_delta <= 0.1 * init_thres:
            #    alpha *= 2  # 增加一个功能，如果sum_delta<=0.0001则*10，如果sum_delta>=0.1则/10
            #elif sum_delta >= 0.1:
            #    alpha /= 2

        else:
            for batch_idx, (data, target) in enumerate(train_loader):
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
                               100. * batch_idx / len(train_loader), loss.item()), file=f)
                    print('threshold:{}'.format(thres), file=f)

        # --------------------软剪枝过程----------------------#

        if chosen_flag == False:
            output_size = return_output(model)
            model, flops, ratio, after_prune_acc, cfg_mask, conv_mask, linear_mask, end_flag = softpruning(f, args, model,thres,end_flag,output_size)
            print('after_prune_acc:{}  prune_ratio:{}  flops_drop:{}'.format(after_prune_acc, ratio, (1 - (flops / flops_init))), file=f)
        else:
            model = do_Mask(f, model, cfg_mask, conv_mask, linear_mask)
            print("Masked! pruned_ratio:{}, flops_drops:{}".format(ratio, (1 - (flops / flops_init))), file=f)
        # 该停止条件是达到了某一个可接受裁剪度
        if end_flag == False and (1 - (flops / flops_init) >= args.exp_flops):
            print("flops drop has already satisfied!")
            end_flag = True
        if after_prune_acc > 0.11 and (ratio >= args.lb or end_flag == True):
            print('The mask has been chosen.', file=f)
            chosen_flag = True
        if after_prune_acc <= 0.11 or ratio > args.ub:
            flag = False
            stop_epoch = epoch

        if flag == True:
            thres_before = thres
            pre_model = copy.deepcopy(model)

        if epoch in [0, 79, 159, 239, 319]:
            index_arr, bn_gammma, index_layer = get_bn_layer(model)
            dataframe = pd.DataFrame({'index': index_arr, 'BN_gamma': bn_gammma, 'layer': index_layer})
            dataframe.to_csv("heatmap/vgg16_{}_{}_{}.csv".format(args.dataset, epoch, args.num), index=False, sep=',')

        print('flag:{},  chosen_flag:{}'.format(flag, chosen_flag),file=f)
        logger.log_value('thres', thres, epoch)
        logger.log_value('after_prune_acc', after_prune_acc, epoch)
        logger.log_value('prune_ratio', ratio, epoch)
        lbfgs_alpha_showtensorboard(args, model, epoch, tb_writer)
        if epoch == args.epochs - 1:
            print("The sparse took", time.time() - t0,file=f)
        # 需要再增加一个进过软剪枝之后的精度结果
        dtype = torch.float
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        prec1 = evaluate(
            net=model,
            data_loader=test_loader,
            dtype=dtype,
            device=device,
        )
        # ————————————————增加一如果剪枝之后，精确度掉到10%的可回滚选项，如果fintune5次回不到正常精度，那就直接回滚到原来的thres
        # ——————————————-——如果软剪枝进程恢复时，应该按照原来剪枝率，计算thres的大小
        if flag == False and ratio >= args.ub and upper_flag == True and prec1 >= 0.11:
            upper_flag = False
        # 这个样子是不行的，思考一下解决方法,把回滚条件搞得再严苛一些
        if flag == False and prec1 > 0.9:
            thres = newthres(model, ratio)
            flag = True
        if flag == False and epoch - stop_epoch >= 10 and (epoch + 9) % 5 == 0 and prec1 <= 0.11:
            thres = thres_before  # 回滚，模型也得回滚
            model = copy.deepcopy(pre_model)
            flag = True
        logger.log_value('test_acc', prec1, epoch)
        tradeoff = args.pec * prec1 + (1 - args.pec) * (1 - (flops / flops_init))
        is_best = tradeoff >= best_tradeoff
        if is_best:
            best_prec1 = prec1
            best_flopd = (1 - (flops / flops_init))
        best_tradeoff = max(tradeoff, best_tradeoff)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': prec1,
            'best_flopd': (1 - (flops / flops_init)),
            'best_tradeoff': tradeoff,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=filepath)

    print("The best tradeoff performance of this round, best_prec1: {}, best_flops: {}".format(best_prec1, best_flopd))
    return best_tradeoff



best_parameters, values, experiment, model = optimize(
    parameters=[
        #{"name": "init_thres", "type": "range", "bounds": [1e-6, 0.1], "log_scale": True},
        #{"name": "x", "type": "range", "bounds": [0.001, 0.1]},
    ],
    evaluation_function=train_evaluate,
    objective_name='accuracy',
    total_trials=10,
)

print(best_parameters)
means, covariances = values
print(means)
print(covariances)

best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])

'''
best_objective_plot = optimization_trace_single_method(
    y=np.maximum.accumulate(best_objectives, axis=1),
    title="Model performance vs. # of iterations",
    ylabel="Tradeoff(prec+ratio), %",
)
render(best_objective_plot)
render(plot_contour(model=model, param_x='init_thres', param_y='x', metric_name='tradeoff'))
'''

data = experiment.fetch_data()
df = data.df
best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
best_arm = experiment.arms_by_name[best_arm_name]
print(best_arm)

train_evaluate(best_arm.parameters)

