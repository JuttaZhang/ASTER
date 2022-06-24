from torch.optim import SGD
import numpy as np

import torch

def prox_l1_norm(w, lamb):
    """perform soft-thresholding on input vector"""
    return torch.from_numpy(np.sign(w) * np.maximum(np.abs(w) - lamb, 0)).cuda()

#regularization=1 设置为默认值,self.prox_frequency=1
class ProxSGD(SGD):

    def __init__(self, params, lr, mu, prox_frequency=1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        super(ProxSGD, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay, nesterov=nesterov)
        self.mu = mu
        # converging: needed to decide whether to load the optimization or the finetune model
        self.converging = False
        self.prox_frequency = prox_frequency
        self.batch = 1

    def step(self, closure=None):
        gamma = 1 #抛除robust
        loss = None
        if closure is not None:
            loss = closure()
        for i, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            if self.converging :
                '''or (not self.converging and i == 0) or \
                    (not self.converging and i == 1 and self.batch % self.prox_frequency != 0):'''
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)# buf = momentum * buf + (1 - dampening) * d_p
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf
                    p.data.add_(-group['lr'], d_p)
            elif not self.converging :
        #and i == 1 and self.batch % self.prox_frequency == 0:
                for j, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    #print(p.grad.data)
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            z = p.data - group['lr'] * p.grad.data
                            z = prox_l1_norm(z.detach().cpu().numpy(), self.regularization*group['lr'])
                            #z = proximal_operator_l1(z, regularization=self.regularization, lr=group['lr']) #soft thresholding
                            buf.mul_(momentum).add_(z - p.data)
                            p.data = z + momentum * buf
                            p.data = gamma*z+ (1-gamma)*p.data
                            # TODO: Is gamma a good value?
                    else:
                        p.data.add_(-group['lr'], d_p)

        return loss