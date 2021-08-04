import sys;
import os;
import numpy as np;
import random;
import torch;
import torch.nn as nn;
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

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, downsample=False, bias=False):
        super(BasicBlock, self).__init__();
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias);
        self.bn1 = nn.BatchNorm2d(out_ch);
        self.relu1 = nn.ReLU(inplace=True);
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias);
        self.bn2 = nn.BatchNorm2d(out_ch);
        self.downsample = None;
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.relu2 = nn.ReLU(inplace=True);

    def forward(self, x):
        identity = x.clone();
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu1(x);
        x = self.bn2(self.conv2(x));
        if self.downsample is not None:
            identity = self.downsample(identity);
        # print(x.shape);
        # print(identity.shape);
        x += identity;
        x = self.relu2(x);

        return x;

class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__();
        self.name = 'Resnet18';
        self.blocks = [2, 2, 2, 2];
        self.in_ch = 64;
        self.conv1 = nn.Conv2d(3, self.in_ch, kernel_size=3, stride=2, padding=3, bias=False);
        self.bn1 = nn.BatchNorm2d(self.in_ch);
        self.relu1 = nn.ReLU(inplace=True);
        self.layer1 = self.make_layers(self.blocks[0], 64);
        self.layer2 = self.make_layers(self.blocks[1], 128, 2);
        self.layer3 = self.make_layers(self.blocks[2], 256, 2);
        self.layer4 = self.make_layers(self.blocks[3], 512, 2);

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1));
        self.fc = nn.Linear(512, num_classes);

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu');
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1);
                nn.init.constant_(m.bias, 0);

    def make_layers(self, blocks, out_ch, stride=1):
        layers = [];
        downsample = False;
        if stride !=1 or self.in_ch != out_ch:
            downsample = True;
        layers.append(BasicBlock(self.in_ch, out_ch, stride=stride, downsample=downsample));
        self.in_ch = out_ch;
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_ch, out_ch));

        return nn.Sequential(*layers);

    def forward(self, x):
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu1(x);
        # x = self.maxpool1(x);

        x = self.layer1(x);
        x = self.layer2(x);
        x = self.layer3(x);
        x = self.layer4(x);

        x = self.avgpool(x);
        x = torch.flatten(x, 1);
        y = self.fc(x);
        # y = self.logsoftmax(y);
        return y;

class BinBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, downsample=False, bias=False):
        super(BinBasicBlock, self).__init__();
        self.conv1 = BinConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias);

        self.conv2 = BinConv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias, activation=False);
        self.downsample = None;
        if downsample:
            self.downsample = nn.Sequential(
                BinConv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=bias, activation=False)
            );
        self.relu = nn.ReLU(inplace=True);

    def forward(self, x):
        identity = x.clone();
        x = self.conv1(x);
        x = self.conv2(x);
        if self.downsample is not None:
            identity = self.downsample(identity);
        # print(x.shape);
        # print(identity.shape);
        x += identity;
        x = self.relu(x);

        return x;

class BinResnet18(nn.Module):
    def __init__(self, num_classes):
        super(BinResnet18, self).__init__();
        self.name = 'BinResnet18';
        self.blocks = [2, 2, 2, 2];
        self.in_ch = 64;
        self.conv1 = nn.Conv2d(3, self.in_ch, kernel_size=3, stride=2, padding=3, bias=False);
        self.bn1 = nn.BatchNorm2d(self.in_ch);
        self.relu1 = nn.ReLU(inplace=True);
        self.layer1 = self.make_layers(self.blocks[0], 64);
        self.layer2 = self.make_layers(self.blocks[1], 128, 2);
        self.layer3 = self.make_layers(self.blocks[2], 256, 2);
        self.layer4 = self.make_layers(self.blocks[3], 512, 2);

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1));
        self.fc = nn.Linear(512, num_classes);
        # self.logsoftmax = nn.LogSoftmax(dim=1);

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu');

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0);

    def make_layers(self, blocks, out_ch, stride=1):
        layers = [];
        downsample = False;
        if stride !=1 or self.in_ch != out_ch:
            downsample = True;
        layers.append(BinBasicBlock(self.in_ch, out_ch, stride=stride, downsample=downsample));
        self.in_ch = out_ch;
        for _ in range(1, blocks):
            layers.append(BinBasicBlock(self.in_ch, out_ch));

        return nn.Sequential(*layers);

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01);

        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu1(x);
        # x = self.maxpool1(x);

        x = self.layer1(x);
        x = self.layer2(x);
        x = self.layer3(x);
        x = self.layer4(x);

        x = self.avgpool(x);
        x = torch.flatten(x, 1);
        y = self.fc(x);
        # y = self.logsoftmax(y);
        return y;

# if __name__ == '__main__':
#     resnet18 = resnet18(10);
#     print(resnet18);
