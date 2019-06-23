import torch
import torch.nn as nn
from utils import compose



class Conv2dbnleaky(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv2dbnleaky, self).__init__()
        self.conv1 = nn.Conv2d(*args, **kwargs)
        self.bn1 = nn.BatchNorm2d(args[1])
        self.relu1 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x

class Conv2dnb(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv2dnb, self).__init__()
        self.nb_kwargs = {"use_bias": False}
        self.nb_kwargs.update(kwargs)
        self.conv1 = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        return x

class Resblock(nn.Module):
    def __init__(self, in_filter, out_filter, n_block):
        super(Resblock, self).__init__()
        self.in_filter = in_filter
        self.out_filter = out_filter
        self.n_block = n_block
        self.conv1 = Conv2dbnleaky(self.in_filter, self.out_filter, 3, stride=2, padding=1)
        self.conv2 = Conv2dbnleaky(self.out_filter, self.out_filter//2, 1)
        self.conv3 = Conv2dbnleaky(self.out_filter//2, self.out_filter//2, 3, padding=1)
        self.conv4 = Conv2dnb(self.out_filter//2, self.out_filter, 1)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(self.n_block):
            y = self.conv2(x)
            y = self.conv3(y)
            y = self.conv4(y)
            x = torch.add(x, y)
        return x

class GlobalAveragePool2d(nn.Module):
    def __init__(self, in_filter, out_filter, kernel_size):
        super(GlobalAveragePool2d, self).__init__()
        self.in_filter = in_filter
        self.out_filter = out_filter
        self.kernel_size = kernel_size
        self.avgpool1 = nn.AvgPool2d(self.kernel_size)
        self.do1 = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(self.in_filter, self.out_filter)

    def forward(self, x):
        x = self.avgpool1(x)
        x = x.view(-1, self.in_filter)
        x = self.do1(x)
        x = self.fc1(x)
        return x

class XBlock(nn.Module):
    def __init__(self, in_filter, out_filter, n_block):
        super(XBlock, self).__init__()
        self.res1 = Resblock(in_filter, out_filter, n_block)
        self.res2 = Resblock(in_filter, out_filter, n_block)
    
    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x)
        return torch.add(x1, x2)

class XBlockv2(nn.Module):
    def __init__(self, in_filter, out_filter, n_block):
        super(XBlockv2, self).__init__()
        self.res1 = Resblock(in_filter, out_filter, n_block)
        self.res2 = Resblock(in_filter, out_filter, n_block)
        self.conv1 = Conv2dbnleaky(in_filter, out_filter, 1)
        self.pool1 = nn.AvgPool2d(2)
    
    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x)
        x3 = self.conv1(x)
        x3 = self.pool1(x3)
        x4 = torch.add(x1, x2)
        x5 = torch.add(x4, x3)
        return x5