import torch;
import torch.nn as nn;
import torch.nn.functional as F

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        # print(size)
        input = input.sign()
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class SimpleConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        x = self.bn(x)
        x = self.activation(x)
        return x

class BinConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, activation=True):
        super().__init__(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(in_ch)
        self.activation = nn.ReLU(inplace=True) if activation else None
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.bn(x)
        x = BinActive.apply(x)
        x = F.conv2d(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        if self.activation is not None:
            x = self.activation(x)
        return x
