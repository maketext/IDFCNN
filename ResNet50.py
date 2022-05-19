import torch
import torch.nn as nn
import random

def initialize_weights(self):
    # track all layers
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform(m.weight)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(x)

def conv7x7(in_planes, out_planes, stride=2, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBL(nn.Module):
    def __init__(self, inplanes, planes, st=1):
        super(BasicBL, self).__init__()
        self.st = st
        self.conv0 = conv1x1(inplanes, out_planes // 4, stride=1)
        self.conv1 = conv1x1(out_planes, out_planes // 4, stride=1)
        self.bn1 = nn.BatchNorm2d(x)
        self.relu1 = Swish()
        self.conv2 = conv3x3(out_planes // 4, out_planes // 4, stride=st)
        self.bn2 = nn.BatchNorm2d(x)
        self.relu2 = Swish()
        self.conv3 = conv1x1(out_planes // 4, out_planes, stride=1)
        self.bn3 = nn.BatchNorm2d(x)
        self.relu3 = Swish()

    def forward(self, init, count):
        out = init
        if st == 2:
            out = self.conv0(out)
        else:
            out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        return out

class MainBL(nn.Module):
    def __init__(self):
        super(MainBL, self).__init__() #224
        self.conv7x7 = conv7x7(3, 64)
        self.mp = nn.MaxPool2d(3, stride=2)
        self.basicBL1 = BasicBL(64, 256, st=1)
        initialize_weights(self)

    def forward(self, x):
        out = x
        N  = [3, 4, 6, 3]
        ST = [[2, 1, 1], [2, 1, 1, 1], [2, 1, 1, 1, 1, 1], [2, 1, 1]]
        SE = [dict(st=64, ed=256), dict(st=256, ed=512), dict(st=512, ed=1024), dict(st=1024, ed=2048)]

        for i, n in enumerate(N):
            for st in enumerate(ST[i]):
                if st == 2:
                    out = BasicBL(SE[i]['st'], SE[i]['ed'], st=st)(out)
                elif st == 1:
                    out = BasicBL(SE[i]['ed'], SE[i]['ed'], st=st)(out)

        return out