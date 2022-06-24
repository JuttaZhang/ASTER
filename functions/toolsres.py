import torch
import sys
sys.path.append('../')
from models.vggs import vgg
from models.resnetex import *

def compute_conv_flops(model: torch.nn.Module, cuda=False) -> float:
    """
    compute the FLOPs for CIFAR models
    NOTE: ONLY compute the FLOPs for Convolution layers and Linear layers
    """

    list_conv = []

    def conv_hook(self, input, output):

        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)

        flops = kernel_ops * output_channels * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        #weight_ops = self.weight.data.ne(0).nelement()
        weight_ops = self.weight.nelement()

        flops = weight_ops
        list_linear.append(flops)

    def add_hooks(net, hook_handles: list):
        """
        apply FLOPs handles to conv layers recursively
        """
        children = list(net.children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                hook_handles.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                hook_handles.append(net.register_forward_hook(linear_hook))
            return
        for c in children:
            add_hooks(c, hook_handles)

    handles = []
    add_hooks(model, handles)
    demo_input = torch.rand(1, 3, 224, 224)
    if cuda:
        demo_input = demo_input.cuda()
        model = model.cuda()
    model(demo_input)

    total_flops = sum(list_conv) + sum(list_linear)

    # clear handles
    for h in handles:
        h.remove()
    return total_flops

checkpoint = torch.load('../get_the_small_model/1018_2_resnet110_1_0.0001_pruned.pth.tar/pruned.pth.tar')
#model_baseline = resnet(depth=56)
#model_baseline = resnet(depth=110)
#model = resnet(depth=56, cfg=checkpoint['cfg'])
model = resnet(depth=110, cfg=checkpoint['cfg'])

model.load_state_dict(checkpoint['state_dict'])
#baseline_flops = compute_conv_flops(model_baseline, cuda=True)
pruned_flops = compute_conv_flops(model, cuda=True)
#print(" --> baseline FLOPs: {}".format(baseline_flops))
print(" --> pruned FLOPs: {}".format(pruned_flops))
#print(" reduced FLOPs ratio: {}".format(1-pruned_flops/6161613440.0))
print("reduced FLOPs ratio: {}".format(1-pruned_flops/12404310656.0))

