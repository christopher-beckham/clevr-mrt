import torch
from torch import nn
from .x import Resnet

class Regressor(Resnet):
    """Use the resnet base in .x and attach a 1D regressor"""
    def __init__(self, *args, **kwargs):
        super(Regressor, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(self.nf*8, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)

    def forward(self, x):
        x = super(Regressor, self).forward(x)
        x = self.fc(x)
        return x

def get_network(n_channels, ndf):
    return Regressor(input_nc=n_channels,
                     nf=ndf)
