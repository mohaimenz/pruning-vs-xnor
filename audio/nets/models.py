import sys;
import os;
import numpy as np;
import random;
import torch;
import torch.nn as nn;
import torch.nn.functional as F;

sys.path.append(os.getcwd());
sys.path.append(os.path.join(os.getcwd(), 'common'));
from common.custom_modules import BinActive, SimpleConv2d, BinConv2d;

#Reproducibility
seed = 42;
random.seed(seed);
np.random.seed(seed);
torch.manual_seed(seed);
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed);
torch.backends.cudnn.deterministic = True;
torch.backends.cudnn.benchmark = False;
###########################################

def get_tfeb_pool_size_component(length):
    # print(length);
    c = [];
    index = 1;
    while index <= 6:
        if length >= 2:
            if index == 6:
                c.append(length);
            else:
                c.append(2);
                length = length // 2;
        else:
           c.append(1);
        index += 1;
    return c;

def get_tfeb_pool_sizes(conv2_ch, width):
    h = get_tfeb_pool_size_component(conv2_ch);
    w = get_tfeb_pool_size_component(width);
    # print(w);
    pool_size = [];
    for  (h1, w1) in zip(h, w):
        pool_size.append((h1, w1));
    return pool_size;

# XNOR implementation of ACDNetV2
#Fundamental:
# Basic Conv layer structure: conv -> batch norm -> relu -> Pool
# XNOR Layer Structure: Batch Norm -> Binarization -> Conv -> Relu -> Pool
# See 'Training XNOR-Networks' in section 3.2 in the original XNOR paper
class BINACDNet(nn.Module):
    def __init__(self, input_length, n_class, sr, ch_conf=None):
        super(BINACDNet, self).__init__();
        self.input_length = input_length;
        self.ch_config = ch_conf;

        stride1 = 2;
        stride2 = 2;
        channels = 8;
        k_size = (3, 3);
        n_frames = (sr/1000)*10; #No of frames per 10ms
        sfeb_pool_size = int(n_frames/(stride1*stride2));
        channels = 8;
        if self.ch_config is None:
            self.ch_config = [channels, channels*8, channels*4, channels*8, channels*8, channels*16, channels*16, channels*32, channels*32, channels*64, channels*64, n_class];
        fcn_no_of_inputs = self.ch_config[-1];
        conv1 = SimpleConv2d(1, self.ch_config[0], (1, 9), (1, stride1));
        conv2 = BinConv2d(self.ch_config[0], self.ch_config[1], (1, 5), (1, stride2));
        conv3 = BinConv2d(1, self.ch_config[2], k_size, padding=1);
        conv4 = BinConv2d(self.ch_config[2], self.ch_config[3], k_size, padding=1);
        conv5 = BinConv2d(self.ch_config[3], self.ch_config[4], k_size, padding=1);
        conv6 = BinConv2d(self.ch_config[4], self.ch_config[5], k_size, padding=1);
        conv7 = BinConv2d(self.ch_config[5], self.ch_config[6], k_size, padding=1);
        conv8 = BinConv2d(self.ch_config[6], self.ch_config[7], k_size, padding=1);
        conv9 = BinConv2d(self.ch_config[7], self.ch_config[8], k_size, padding=1);
        conv10 = BinConv2d(self.ch_config[8], self.ch_config[9], k_size, padding=1);
        conv11 = BinConv2d(self.ch_config[9], self.ch_config[10], k_size, padding=1);
        conv12 = BinConv2d(self.ch_config[10], self.ch_config[11], (1, 1));

        n_frames = (sr/1000)*10; #No of frames per 10ms
        sfeb_pool_size = int(n_frames/(stride1*stride2));
        self.sfeb = nn.Sequential(
            conv1,
            conv2,
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        );

        tfeb_modules = [];
        self.tfeb_width = int(((self.input_length / sr)*1000)/10); # 10ms frames of audio length in seconds
        tfeb_pool_sizes = get_tfeb_pool_sizes(self.ch_config[1], self.tfeb_width);
        p_index = 0;
        for i in [3,4,6,8,10]:
            tfeb_modules.append(eval('conv{}'.format(i)));
            if i != 3:
                tfeb_modules.append(eval('conv{}'.format(i+1)));

            h, w = tfeb_pool_sizes[p_index];
            if h>1 or w>1:
                tfeb_modules.append(nn.MaxPool2d(kernel_size = (h,w)));
            p_index += 1;

        tfeb_modules.append(nn.Dropout(0.2, inplace=False));
        tfeb_modules.append(conv12);
        h, w = tfeb_pool_sizes[-1];
        if h>1 or w>1:
            tfeb_modules.append(nn.AvgPool2d(kernel_size = (h,w)));

        self.tfeb = nn.Sequential(*tfeb_modules);

        fcn = nn.Linear(fcn_no_of_inputs, n_class, bias=True);
        fcn.apply(lambda m: nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid'));
        self.output = nn.Sequential(
            nn.Flatten(),
            fcn,
            nn.Softmax(dim=1)
        );

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        x = self.sfeb(x);
        x = x.permute((0, 2, 1, 3));
        x = self.tfeb(x);
        y = self.output(x);
        return y;

#required for quantization:
from torch.quantization import QuantStub, DeQuantStub
class ACDNet(nn.Module):
    def __init__(self, input_length, n_class, sr, ch_conf=None, quantize = False):
        super(ACDNet, self).__init__();
        self.input_length = input_length;
        self.ch_config = ch_conf;
        self.quantize = quantize;

        stride1 = 2;
        stride2 = 2;
        channels = 8;
        k_size = (3, 3);
        n_frames = (sr/1000)*10; #No of frames per 10ms

        sfeb_pool_size = int(n_frames/(stride1*stride2));
        # tfeb_pool_size = (2,2);
        if self.ch_config is None:
            self.ch_config = [channels, channels*8, channels*4, channels*8, channels*8, channels*16, channels*16, channels*32, channels*32, channels*64, channels*64, n_class];
        # avg_pool_kernel_size = (1,4) if self.ch_config[1] < 64 else (2,4);
        fcn_no_of_inputs = self.ch_config[-1];
        conv1, bn1 = self.make_layers(1, self.ch_config[0], (1, 9), (1, stride1));
        conv2, bn2 = self.make_layers(self.ch_config[0], self.ch_config[1], (1, 5), (1, stride2));
        conv3, bn3 = self.make_layers(1, self.ch_config[2], k_size, padding=1);
        conv4, bn4 = self.make_layers(self.ch_config[2], self.ch_config[3], k_size, padding=1);
        conv5, bn5 = self.make_layers(self.ch_config[3], self.ch_config[4], k_size, padding=1);
        conv6, bn6 = self.make_layers(self.ch_config[4], self.ch_config[5], k_size, padding=1);
        conv7, bn7 = self.make_layers(self.ch_config[5], self.ch_config[6], k_size, padding=1);
        conv8, bn8 = self.make_layers(self.ch_config[6], self.ch_config[7], k_size, padding=1);
        conv9, bn9 = self.make_layers(self.ch_config[7], self.ch_config[8], k_size, padding=1);
        conv10, bn10 = self.make_layers(self.ch_config[8], self.ch_config[9], k_size, padding=1);
        conv11, bn11 = self.make_layers(self.ch_config[9], self.ch_config[10], k_size, padding=1);
        conv12, bn12 = self.make_layers(self.ch_config[10], self.ch_config[11], (1, 1));
        fcn = nn.Linear(fcn_no_of_inputs, n_class);
        nn.init.kaiming_normal_(fcn.weight, nonlinearity='sigmoid') # kaiming with sigoid is equivalent to lecun_normal in keras

        self.sfeb = nn.Sequential(
            #Start: Filter bank
            conv1, bn1, nn.ReLU(),\
            conv2, bn2, nn.ReLU(),\
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        );

        tfeb_modules = [];
        self.tfeb_width = int(((self.input_length / sr)*1000)/10); # 10ms frames of audio length in seconds
        tfeb_pool_sizes = get_tfeb_pool_sizes(self.ch_config[1], self.tfeb_width);
        p_index = 0;
        for i in [3,4,6,8,10]:
            tfeb_modules.extend([eval('conv{}'.format(i)), eval('bn{}'.format(i)), nn.ReLU()]);

            if i != 3:
                tfeb_modules.extend([eval('conv{}'.format(i+1)), eval('bn{}'.format(i+1)), nn.ReLU()]);

            h, w = tfeb_pool_sizes[p_index];
            if h>1 or w>1:
                tfeb_modules.append(nn.MaxPool2d(kernel_size = (h,w)));
            p_index += 1;

        tfeb_modules.append(nn.Dropout(0.2));
        tfeb_modules.extend([conv12, bn12, nn.ReLU()]);
        h, w = tfeb_pool_sizes[-1];
        if h>1 or w>1:
            tfeb_modules.append(nn.AvgPool2d(kernel_size = (h,w)));
        tfeb_modules.extend([nn.Flatten(), fcn]);

        self.tfeb = nn.Sequential(*tfeb_modules);

        self.output = nn.Sequential(
            nn.Softmax(dim=1)
        );

        if self.quantize:
            self.quant = QuantStub();
            self.dequant = DeQuantStub();

    def forward(self, x):
        if self.quantize:
            #Quantize input
            x = self.quant(x);

        x = self.sfeb(x);
        #swapaxes
        x = x.permute((0, 2, 1, 3));
        x = self.tfeb(x);

        if self.quantize:
            #DeQuantize features before feeding to softmax
            x = self.dequant(x);
        y = self.output[0](x);
        return y;

    def make_layers(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=0, bias=False):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias);
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); # kaiming with relu is equivalent to he_normal in keras
        bn = nn.BatchNorm2d(out_channels);
        return conv, bn;

def GetBINACDNetModel(input_len=66650, nclass=50, sr=44100, channel_config=None):
    net = BINACDNet(input_len, nclass, sr, ch_conf=channel_config);
    return net;
def GetACDNetModel(input_len=66650, nclass=50, sr=44100, channel_config=None, quantize=False):
    net = ACDNet(input_len, nclass, sr, channel_config, quantize);
    return net;
