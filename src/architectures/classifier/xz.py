import numpy as np
import torch
from torch import nn
from .core import (FirstResBlockXZ,
                   ResBlockXZ)

class Resnet(nn.Module):
    """ResNet classifier which conditions on x and z"""
    def __init__(self,
                 nf,
                 z_dim,
                 input_nc=3,
                 n_out=1):
        super(Resnet, self).__init__()

        self.rb1 = FirstResBlockXZ(input_nc, nf, stride=2, z_dim=z_dim)
        self.rb2 = ResBlockXZ(nf, nf*2, stride=2, z_dim=z_dim)
        self.rb3 = ResBlockXZ(nf*2, nf*4, stride=2, z_dim=z_dim)
        self.rb4 = ResBlockXZ(nf*4, nf*8, stride=2, z_dim=z_dim)
        self.rb5 = ResBlockXZ(nf*8, nf*8, z_dim=z_dim)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(4)
        self.nf = nf

        self.fc = nn.Linear(nf*8, n_out)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)

    def forward(self, x, z):
        x = self.rb1(x, z)
        x = self.rb2(x, z)
        x = self.rb3(x, z)
        x = self.rb4(x, z)
        x = self.rb5(x, z)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, x.size(1))
        result = self.fc(x)
        return result

def get_network(n_channels, ndf):
    return Resnet(input_nc=n_channels,
                  z_dim=3,
                  nf=ndf,
                  n_out=100)
