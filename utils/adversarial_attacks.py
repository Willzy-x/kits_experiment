import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils.utils import where, cuda

class FGSM_Attack(object):

    # default cirterion = F.CrossEntropy()
    def __init__(self, net, criterion):
      self.net = net
      self.criterion = criterion

    def fgsm(self, x, y, targeted=False, eps=0.03, x_val_min=-79, x_val_max=304, softmax=None):
        x_adv = Variable(x.data, requires_grad=True)
        h_adv = self.net(x_adv)
        if softmax is not None:
            h_adv = softmax(h_adv, dim=1)

        if targeted:
            cost = self.criterion(h_adv, y[:, 0, :, :, :])
        else:
            cost = -self.criterion(h_adv, y[:, 0, :, :, :])

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

        h = self.net(x)
        h_adv = self.net(x_adv)
        if softmax is not None:
                h_adv = softmax(h_adv, dim=1)
                h = softmax(h, dim=1)

        return x_adv, h_adv, h

    def i_fgsm(self, x, y, targeted=False, eps=0.03, alpha=1, iteration=1, x_val_min=-79, x_val_max=304, softmax=None):
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            h_adv = self.net(x_adv)
            if softmax is not None:
                h_adv = softmax(h_adv, dim=1)
            if targeted:
                cost = self.criterion(h_adv, y[:, 0, :, :, :])
            else:
                cost = -self.criterion(h_adv, y[:, 0, :, :, :])

            self.net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            x_adv.grad.sign_()
            x_adv = x_adv - alpha*x_adv.grad
            x_adv = where(x_adv > x+eps, x+eps, x_adv)
            x_adv = where(x_adv < x-eps, x-eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        h = self.net(x)
        h_adv = self.net(x_adv)
        if softmax is not None:
                h_adv = softmax(h_adv, dim=1)
                h = softmax(h, dim=1)

        return x_adv, h_adv, h
