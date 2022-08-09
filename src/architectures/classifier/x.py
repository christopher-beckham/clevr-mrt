import numpy as np
import torch
from torch import nn
from .core import (FirstResBlock,
                   ResBlock)

class ResnetBase(nn.Module):
    """ResNet core which conditions on x"""
    def __init__(self,
                 nf,
                 input_nc=3):
        super(ResnetBase, self).__init__()

        self.rb1 = FirstResBlock(input_nc, nf, stride=2, norm='batch')
        self.rb2 = ResBlock(nf, nf*2, stride=2, norm='batch')
        self.rb3 = ResBlock(nf*2, nf*4, stride=2, norm='batch')
        self.rb4 = ResBlock(nf*4, nf*8, stride=2, norm='batch')
        self.rb5 = ResBlock(nf*8, nf*8, norm='batch')
        self.relu = nn.ReLU()
        #self.pool = nn.AvgPool2d(4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.nf = nf

    def embed(self, x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, x.size(1))
        return x

    def forward(self, x):
        raise NotImplementedError()

#def get_network(n_channels, ndf):
#    return Resnet(input_nc=n_channels,
#                  z_dim=3,
#                  nf=ndf,
#                  n_out=100)
