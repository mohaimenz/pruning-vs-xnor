import torch.nn as nn
import numpy

class BinOp():
    def __init__(self, model):
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if type(m).__name__ in ['BinConv2d', 'BinLinear']:
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp)
                self.target_modules.append(m.weight)

        self.num_of_params = len(self.target_modules)
        return

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            # s[1] holds the no of input channel. The BinConv with input channel=1 (e.g: conv3) should \
            # not go through this mean center process to avoid null matrix scenario.
            if s[1] > 1:
                negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                        mul(-1).expand_as(self.target_modules[index].data)
                self.target_modules[index].data.add_(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.clamp_(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            if len(s) == 4:
                m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            elif len(s) == 2:
                m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
            self.target_modules[index].data = \
                    self.target_modules[index].data.sign().mul(m.expand(s))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            m = m.clone()
            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
