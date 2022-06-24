import sys
sys.path.append('../')
from models.resnet import *
from torchscope import scope
#from torchvision.models import vgg16_bn
from models.vggs import vgg
import torch

##x = x.view(-1, 512)


checkpoint = torch.load('../get_the_small_model/pruned_925_vgg16_6_0.001.pth.tar/pruned.pth.tar')
model = vgg(dataset='cifar10', depth=16, cfg=checkpoint['cfg'])
model.load_state_dict(checkpoint['state_dict'])
print(model)
scope(model, input_size=(3, 224, 224))
