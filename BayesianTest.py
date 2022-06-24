import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from models.vggs import vgg

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from torch.utils.data import DataLoader, Dataset, Subset
from ax.utils.tutorials.cnn_utils import evaluate

parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 160)')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

def test(model,test_loader):
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
    '''print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),file=f)'''
    return correct / float(len(test_loader.dataset))

def train_evaluate(parameterization):
    # constructing a new training data loader allows us to tune the batch size
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


    # Get neural net
    net = vgg(depth=16, dataset='cifar10')
    net.cuda()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),  # or any optimizer you prefer
                          lr=0.001,  # 0.001 is used if no lr is specified
                          momentum= 0.9
                          )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=parameterization.get("gamma", 1.0),  # default is no learning rate decay
    )

    # num_epochs = parameters.get("num_epochs", 3) # Play around with epoch number
    # Train Network
    for _ in range(args.epochs):
        for inputs, labels in train_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
    total = 0  # out_channel数量
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    bn = torch.zeros(total)  # 把所以bn层的｜weight｜合到同一个tensor中
    index = 0
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size
    y, i = torch.sort(bn)  # y表示排序后的原tensor，i表示排在第几位
    y = y.cuda()
    thre_index = 0
    for i in range(len(y)):
        if abs(y[i]) < parameterization.get("thres", 0.1):
            thre_index += 1
    ratio = thre_index / total
    prec1 = evaluate(
        net=net,
        data_loader=test_loader,
        dtype=dtype,
        device=device,
    )

    acc = prec1+ratio
    # return the accuracy of the model as it was trained in this run
    return acc


# torch.cuda.set_device(0) #this is sometimes necessary for me
dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_parameters, values, experiment, model = optimize(
    parameters=[
        #{"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        #{"name": "batchsize", "type": "range", "bounds": [16, 128]},
        {"name": "thres", "type": "range", "bounds": [0.0, 1.0]},
        # {"name": "max_epoch", "type": "range", "bounds": [1, 30]},
        # {"name": "stepsize", "type": "range", "bounds": [20, 40]},
    ],

    evaluation_function=train_evaluate,
    objective_name='acc',
)

print(best_parameters)
means, covariances = values
print(means)
print(covariances)

best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])

best_objective_plot = optimization_trace_single_method(
    y=np.maximum.accumulate(best_objectives, axis=1),
    title="Model performance vs. # of iterations",
    ylabel="Classification Accuracy, %",
)
render(best_objective_plot)

render(plot_contour(model=model, param_x='batchsize', param_y='lr', metric_name='accuracy'))

data = experiment.fetch_data()
df = data.df
best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
best_arm = experiment.arms_by_name[best_arm_name]
best_arm

