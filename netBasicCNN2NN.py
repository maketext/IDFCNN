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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def upSample(inp, scale):
    return nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)(inp)


def gating(inp, scale=1):
    B, C, H, W = inp.shape  # BCHW
    z = torch.zeros([1, H, W], dtype=torch.float32)
    o = torch.ones([1, H, W], dtype=torch.float32)
    CHW = torch.tensor([])
    for i in range(C):
        if i % 2 == 0:
            CHW = torch.cat((CHW, z), dim=0)
        elif i % 2 == 1:
            CHW = torch.cat((CHW, o), dim=0)
    CHW = CHW.unsqueeze(dim=0)
    gate = torch.tensor([])
    for i in range(B):
        gate = torch.cat((gate, CHW), dim=0)
    inp = (inp * gate)
    inp = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)(inp)
    return inp


def gatingRandom(inp, scale=1):
    B, C, H, W = inp.shape  # BCHW
    z = torch.zeros([1, H, W], dtype=torch.float32)
    o = torch.ones([1, H, W], dtype=torch.float32)
    gate = torch.tensor([])
    for i in range(B):

        CHW = torch.tensor([])
        for i in range(C):
            r = random.random()
            if r < 0.5:
                CHW = torch.cat((CHW, z), dim=0)
            else:
                CHW = torch.cat((CHW, o), dim=0)
        CHW = CHW.unsqueeze(dim=0)

        gate = torch.cat((gate, CHW), dim=0)
    inp = (inp * gate)
    inp = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)(inp)
    return inp

class NNBL(nn.Module):
    def __init__(self, planes, outplanes):
        super(NNBL, self).__init__()

        self.nn1 = nn.Linear(planes, planes)
        self.relu1 = Swish()
        self.nn2 = nn.Linear(planes, outplanes)
        self.relu2 = Swish()
        self.nn3 = nn.Linear(outplanes, outplanes)

    def forward(self, x):
        out = self.nn1(x)
        out = self.relu1(out)
        out = self.nn2(out)
        out = self.relu2(out)
        out = self.nn3(out)
        return out

class GhostBL(nn.Module):
    def __init__(self, planes):
        super(GhostBL, self).__init__()
        self.conv1 = conv3x3(planes, planes, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = Swish()
        self.conv2 = conv1x1(planes*2, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = Swish()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = torch.cat([x, out], dim=1)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out



class BasicBL(nn.Module):
    def __init__(self, inplanes, planes, st=1):
        super(BasicBL, self).__init__()
        x = planes - inplanes // 2
        self.st = st
        self.conv1 = conv1x1(inplanes // 2, x, stride=1)
        self.bn1 = nn.BatchNorm2d(x)
        self.relu1 = Swish()
        self.conv2 = conv3x3(x, x, stride=st)
        self.bn2 = nn.BatchNorm2d(x)
        self.relu2 = Swish()
        self.conv3 = conv1x1(x, x, stride=1)
        self.bn3 = nn.BatchNorm2d(x)
        self.relu3 = Swish()

    def forward(self, init):
        #print(f'x.size={x.shape}')
        sp = init.shape[1]//2
        out1, out2 = torch.split(init, split_size_or_sections=[sp, sp], dim=1)

        out1 = self.conv1(out1)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu2(out1)
        out1 = self.conv3(out1)
        out1 = self.bn3(out1)
        out1 = self.relu3(out1)

        out2 = upSample(out2, 1/self.st)

        return torch.cat([out1, out2], dim=1)


class MainBL(nn.Module):
    def __init__(self):
        super(MainBL, self).__init__() #224
        self.basicBL1 = conv3x3(3, 8) #224 112
        self.bottleBL10 = BasicBL(8, 16, st=2) #112 56
        self.bottleBL11 = BasicBL(16, 32)
        self.bottleBL12 = BasicBL(32, 32)
        self.GhostBL1 = GhostBL(32+3+3) #112

        self.basicBL2 = BasicBL(32+3+3, 48, st=2) #56 28
        self.bottleBL20 = BasicBL(48, 64)
        self.bottleBL21 = BasicBL(64, 64)
        self.bottleBL22 = BasicBL(64, 64) #56
        self.GhostBL2 = GhostBL(64)

        self.basicBL3 = BasicBL(64, 80, st=2)
        #self.bottleBL30 = BasicBL(80, 80)
        #self.bottleBL31 = BasicBL(80, 80)
        self.bottleBL32 = BasicBL(80, 100) #28 14
        self.GhostBL3 = GhostBL(100)

        #self.basicBL4 = BasicBL(100+8, 128, st=2)
        #self.bottleBL40 = BasicBL(128, 128)
        #self.bottleBL41 = BasicBL(128, 128)
        #self.bottleBL42 = BasicBL(128, 138) #14 7
        #self.GhostBL4 = GhostBL(138)

        self.avg = nn.AvgPool2d(14)
        self.conv1 = conv1x1(206, 100)
        self.conv11 = conv1x1(100, 50)
        self.conv111 = conv1x1(50, 5)
        self.conv28_8 = conv1x1(64, 8)

        self.nn1 = NNBL(8*28*28, 2000)
        self.nn2 = NNBL(2000, 1200)
        self.nn3 = NNBL(1200, 1*28*28)
        initialize_weights(self)

    def forward(self, x):
        out = upSample(x, 0.5)
        raw = upSample(x, 0.25)

        out = self.basicBL1(out)
        out = self.bottleBL10(out)
        out = self.bottleBL11(out) #32
        out = torch.cat([raw, out], dim=1) #35
        out = torch.cat([raw, out], dim=1) #38

        #out = self.bottleBL12(out)
        out1 = self.GhostBL1(out) # 112x8x8

        out = self.basicBL2(out1)
        out = self.bottleBL20(out)
        #out = self.bottleBL21(out)
        #out = self.bottleBL22(out)
        out2 = self.GhostBL2(out) # 56x32x32




        par = self.conv28_8(out)
        par = self.nn1(par.view(par.shape[0], -1))
        par = self.nn2(par)
        par = self.nn3(par)

        out = self.basicBL3(out2) #64*28*28
        #out = self.bottleBL30(out)
        #out = self.bottleBL31(out)
        out = self.bottleBL32(out)
        out3 = self.GhostBL3(out)
        out3 = torch.cat([par.view(par.shape[0],-1,14,14) * 0.1, out3], dim=1)


        #out = self.basicBL4(out3)
        #out = self.bottleBL40(out)
        #out = self.bottleBL41(out)
        #out = self.bottleBL42(out)
        #out4 = self.GhostBL4(out) # 14

        #out4 = upSample(out4, 2) #100 56
        out3 = upSample(out3, 1) #100 28
        out2 = upSample(out2, 0.5) #64 28
        out1 = upSample(out1, 0.25) #32 28
        out = torch.cat([out1, out2, out3], dim=1)

        #out = torch.cat([out1, out2, out3, out4], dim=1)

        out = self.avg(out) #93 203 14 14
        out = self.conv1(out)
        out = self.conv11(out)
        out = self.conv111(out)

        out = out.squeeze(2).squeeze(2)


        #out[:,4:8] = self.softmax(out[:, 4:8])
        #out = torch.cat([out[:, :4], self.softmax(out[:, 4:8])], dim=1)
        return out