import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def _split(z):
    len_ = z.size(1)
    scale = z[:, 0:(len_//2)]
    shift = z[:, (len_//2):]
    return scale, shift

def _rshp(z):
    return z.view(-1, z.size(1), 1, 1)

# Network definitions for when one is conditioning
# on both x and z

class ResBlockXZ(nn.Module):

    def __init__(self, in_channels, out_channels, z_dim, stride=1):
        super(ResBlockXZ, self).__init__()

        self.norm = nn.InstanceNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.relu = nn.ReLU()
        if stride == 1:
            self.pool = None
        else:
            self.pool = nn.AvgPool2d(2, stride=stride, padding=0)

        self.mlp = nn.Linear(z_dim, in_channels*2)

        #if stride == 1:
        #    self.model = nn.Sequential(
        #        nn.ReLU(),
        #        self.conv1,
        #        nn.ReLU(),
        #        self.conv2
        #    )
        #else:
        #    self.model = nn.Sequential(
        #        nn.ReLU(),
        #        self.conv1,
        #        nn.ReLU(),
        #        self.conv2,
        #        nn.AvgPool2d(2, stride=stride, padding=0)
        #    )
        self.bypass = nn.Sequential()
        if in_channels != out_channels:
            self.bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass.weight.data, np.sqrt(2))
        if stride != 1:
            self.bypass = nn.Sequential(
                self.bypass,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x, z):
        #return self.model(x) + self.bypass(x)
        h = x
        h = self.relu(h)
        h = self.norm(h)
        scale, shift = _split(_rshp(self.mlp(z)))
        h = scale*h + shift
        h = self.conv1(h)
        h = self.relu(h)
        h = self.conv2(h)
        if self.pool is not None:
            h = self.pool(h)
        return h + self.bypass(x)

class FirstResBlockXZ(nn.Module):

    def __init__(self, in_channels, out_channels, z_dim, stride=1):
        super(FirstResBlockXZ, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        self.mlp = nn.Linear(z_dim, out_channels*2)

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2)

        #self.model = nn.Sequential(
        #    self.conv1,
        #    nn.ReLU(),
        #    self.conv2,
        #    nn.AvgPool2d(2)
        #    )
        #self.bypass = nn.Sequential(
        #    nn.AvgPool2d(2),
        #    self.bypass_conv,
        #)

    def forward(self, x, z):
        h = x
        #return self.model(x) + self.bypass(x)
        h = self.conv1(h)
        scale, shift = _split(_rshp(self.mlp(z)))
        h = scale*self.norm(h) + shift
        h = self.relu(h)
        h = self.pool(h)
        return h + self.pool(self.bypass_conv(x))

##########################

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm='instance'):
        assert norm in ['instance', 'batch']

        super(ResBlock, self).__init__()

        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(in_channels, affine=True)
        else:
            self.norm = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.relu = nn.ReLU()
        if stride == 1:
            self.pool = None
        else:
            self.pool = nn.AvgPool2d(2, stride=stride, padding=0)

        self.bypass = nn.Sequential()
        if in_channels != out_channels:
            self.bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass.weight.data, np.sqrt(2))
        if stride != 1:
            self.bypass = nn.Sequential(
                self.bypass,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        h = x
        h = self.relu(h)
        h = self.norm(h)
        h = self.conv1(h)
        h = self.relu(h)
        h = self.conv2(h)
        if self.pool is not None:
            h = self.pool(h)
        return h + self.bypass(x)

class FirstResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm='instance'):
        assert norm in ['instance', 'batch']

        super(FirstResBlock, self).__init__()

        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(in_channels, affine=True)
        else:
            self.norm = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.norm(h)
        h = self.relu(h)
        h = self.pool(h)
        return h + self.pool(self.bypass_conv(x))
