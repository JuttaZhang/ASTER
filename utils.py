import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import copy
from tensorboardX import SummaryWriter

def find_upper_bound(model):
    upper_bound = 1
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            temp,_ = m.weight.data.topk(k=2, largest=True, sorted=True)
            y, _ = torch.sort(temp, descending=False)
            temp = y[0]
            upper_bound = temp if temp < upper_bound else upper_bound
    return upper_bound

def find_upper_bound_layer(model):
    upper_bound = torch.ones(55)
    print(upper_bound)
    i = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            temp,_ = m.weight.data.topk(k=int(0.0625*m.weight.data.numel())+1, largest=True, sorted=True)
            y, _ = torch.sort(temp, descending=False)
            temp = y[0]
            upper_bound[i] = temp if temp < upper_bound[i] else upper_bound[i]
            i += 1
            print(upper_bound[i])
    return upper_bound

def count_total(model):
    total = 0  # out_channel数量
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    return total

def prune_ratio(model):
    total=count_total(model)
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size
    y, i = torch.sort(bn)  # y表示排序后的原tensor，i表示排在第几位
    thre_index = 0
    y = y.cuda()
    for i in range(len(y)):
        if abs(y[i]) == 0:
            thre_index += 1
    prune_ratio = thre_index / total
    return prune_ratio
def newthres(model,ratio):
    total = count_total(model)
    bn = torch.zeros(total)
    index = 0
    thres = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size
    bn = torch.abs(bn)
    y, i = torch.sort(bn)  # y表示排序后的原tensor，i表示排在第几位
    y = y.cuda()
    bound = int(ratio*total+1)
    thres = y[bound]
    '''for i in range(len(y)):
        if abs(y[i]) != 0:
            thres = y[i]
            break'''
    return thres
def get_bnparameters(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            yield m.weight

def gensubgradient(model):
    sub=[]
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            temp = torch.zeros_like(m.weight.grad.data).cuda()
            temp.add_(torch.sign(m.weight.data))
            sub.append(temp)
    return sub
def gensubgradientbylayer(model):
    sub=[]
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            temp = torch.zeros_like(m.weight.grad.data).cpu()
            temp.add_(torch.sign(m.weight.data).cpu())
            sub.append(temp)
    return sub
"""
    feature.1.weight
    'feature.1.bias'
    'feature.1.running_mean'
    'feature.1.running_var'
    'feature.1.num_batches_tracked'这个跟weight不是同维度的
"""
def dropzero(model, sub): # 去除subgradient为0的向量
    tempmodel = copy.deepcopy(model)


    for i in range(len(sub)):
        for j in range(len(sub[i])):
            if sub[i][j] == 0:
                print("1")

    return

def get_bn(model):
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
    return bn
def showtensorboard_by_layer(args,model,epoch,tb_writer):
    bn_weights = get_bn(model)
    tb_writer.add_histogram('prunebylayer_{}_{}_{}'.format(args.model, args.num, args.s), bn_weights.numpy(), epoch,
                            bins='doane')

def showtensorboard(args,model,epoch,tb_writer):
    bn_weights = get_bn(model)
    tb_writer.add_histogram('{}_{}_{}_{}_{}'.format(args.model, args.s,args.test_sample_num,args.r,args.recursion_depth), bn_weights.numpy(), epoch, bins='doane')

def oldshowtensorboard(args,model,epoch,tb_writer):
    bn_weights = get_bn(model)
    tb_writer.add_histogram('old_{}_{}_{}'.format(args.model, args.num, args.s), bn_weights.numpy(), epoch, bins='doane')

def lbfgs_alpha_showtensorboard(args,model,epoch,tb_writer):
    bn_weights = get_bn(model)
    tb_writer.add_histogram('lbfgs_alpha_0.85_{}_{}_{}'.format(args.model, args.num, args.s), bn_weights.numpy(), epoch, bins='doane')
#%% Compute Statistics (Training Loss, Test Loss, Test Accuracy)

def prox_lbfgs_alpha_showtensorboard(args,model,epoch,tb_writer):
    bn_weights = get_bn(model)
    tb_writer.add_histogram('prox_lbfgs_alpha_0.85_{}_{}_{}'.format(args.model, args.num, args.s), bn_weights.numpy(), epoch, bins='doane')

def only_prox_showtensorboard(args,model,epoch,tb_writer):
    bn_weights = get_bn(model)
    tb_writer.add_histogram('only_prox_0.85_{}_{}_{}'.format(args.model, args.num, args.s), bn_weights.numpy(), epoch, bins='doane')

def compute_stats(X_train, y_train, X_test, y_test, opfun, accfun, ghost_batch=128):
    """
    Computes training loss, test loss, and test accuracy efficiently.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 8/29/18.

    Inputs:
        X_train (nparray): set of training examples
        y_train (nparray): set of training labels
        X_test (nparray): set of test examples
        y_test (nparray): set of test labels
        opfun (callable): computes forward pass over network over sample Sk
        accfun (callable): computes accuracy against labels
        ghost_batch (int): maximum size of effective batch (default: 128)

    Output:
        train_loss (float): training loss
        test_loss (float): test loss
        test_acc (float): test accuracy

    """

    # compute test loss and test accuracy
    test_loss = 0
    test_acc = 0

    # loop through test data
    for smpl in np.array_split(np.random.permutation(range(X_test.shape[0])), int(X_test.shape[0]/ghost_batch)):

        # define test set targets
        if(torch.cuda.is_available()):
            test_tgts = torch.from_numpy(y_test[smpl]).cuda().long().squeeze()
        else:
            test_tgts = torch.from_numpy(y_test[smpl]).long().squeeze()

        # define test set ops
        testops = opfun(X_test[smpl])

        # accumulate weighted test loss and test accuracy
        if(torch.cuda.is_available()):
            test_loss += F.cross_entropy(testops, test_tgts).cpu().item()*(len(smpl)/X_test.shape[0])
        else:
            test_loss += F.cross_entropy(testops, test_tgts).item()*(len(smpl)/X_test.shape[0])

        test_acc += accfun(testops, y_test[smpl])*(len(smpl)/X_test.shape[0])

    # compute training loss
    train_loss = 0

    # loop through training data
    for smpl in np.array_split(np.random.permutation(range(X_train.shape[0])), int(X_test.shape[0]/ghost_batch)):

        # define training set targets
        if(torch.cuda.is_available()):
            train_tgts = torch.from_numpy(y_train[smpl]).cuda().long().squeeze()
        else:
            train_tgts = torch.from_numpy(y_train[smpl]).long().squeeze()

        # define training set ops
        trainops = opfun(X_train[smpl])

        # accumulate weighted training loss
        if(torch.cuda.is_available()):
            train_loss += F.cross_entropy(trainops, train_tgts).cpu().item()*(len(smpl)/X_train.shape[0])
        else:
            train_loss += F.cross_entropy(trainops, train_tgts).item()*(len(smpl)/X_train.shape[0])

    return train_loss, test_loss, test_acc


#%% Compute Objective and Gradient Helper Function

def get_grad(optimizer, X_Sk, y_Sk, opfun, ghost_batch=128):
    """
    Computes objective and gradient of neural network over data sample.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 8/29/18.

    Inputs:
        optimizer (Optimizer): the PBQN optimizer
        X_Sk (nparray): set of training examples over sample Sk
        y_Sk (nparray): set of training labels over sample Sk
        opfun (callable): computes forward pass over network over sample Sk
        ghost_batch (int): maximum size of effective batch (default: 128)

    Outputs:
        grad (tensor): stochastic gradient over sample Sk
        obj (tensor): stochastic function value over sample Sk

    """

    if(torch.cuda.is_available()):
        obj = torch.tensor(0, dtype=torch.float).cuda()
    else:
        obj = torch.tensor(0, dtype=torch.float)

    Sk_size = X_Sk.shape[0]

    optimizer.zero_grad()

    # loop through relevant data
    for idx in np.array_split(np.arange(Sk_size), max(int(Sk_size/ghost_batch), 1)):

        # define ops
        ops = opfun(X_Sk[idx])

        # define targets
        if(torch.cuda.is_available()):
            tgts = Variable(torch.from_numpy(y_Sk[idx]).cuda().long().squeeze())
        else:
            tgts = Variable(torch.from_numpy(y_Sk[idx]).long().squeeze())

        # define loss and perform forward-backward pass
        loss_fn = F.cross_entropy(ops, tgts)*(len(idx)/Sk_size)
        loss_fn.backward()

        # accumulate loss
        obj += loss_fn

    # gather flat gradient
    grad = optimizer._gather_flat_grad()

    return grad, obj

#%% Adjusts Learning Rate Helper Function

def adjust_learning_rate(optimizer, learning_rate):
    """
    Sets the learning rate of optimizer.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 8/29/18.

    Inputs:
        optimizer (Optimizer): any optimizer
        learning_rate (float): desired steplength

    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    return

#%% CUTEst PyTorch Interface
    
class CUTEstFunction(torch.autograd.Function):
    """
    Converts CUTEst problem using PyCUTEst to PyTorch function.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 9/21/18.

    """

    @staticmethod
    def forward(ctx, input, problem):
        x = input.clone().detach().numpy()
        obj, grad = problem.obj(x, gradient=True)
        ctx.save_for_backward(torch.tensor(grad, dtype=torch.float))
        return torch.tensor(obj, dtype=torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad, None

class CUTEstProblem(torch.nn.Module):
    """
    Converts CUTEst problem to torch neural network module.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 9/21/18.

    Inputs:
        problem (callable): CUTEst problem interfaced through PyCUTEst

    """

    def __init__(self, problem):
        super(CUTEstProblem, self).__init__()
        # get initialization
        x = torch.tensor(problem.x0, dtype=torch.float)
        x.requires_grad_()

        # store variables and problem
        self.variables = torch.nn.Parameter(x)
        self.problem = problem

    def forward(self):
        model = CUTEstFunction.apply
        return model(self.variables, self.problem)

    def grad(self):
        return self.variables.grad

    def x(self):
        return self.variables
