import torch
from torch import nn
from .xz import Resnet

class ResnetBase(Resnet):
    """Triplet resnet which conditions on x and z"""
    def __init__(self, *args, **kwargs):
        super(ResnetBase, self).__init__(*args, **kwargs)
        del self.fc

    def forward(self, x, z):
        x = self.rb1(x, z)
        x = self.rb2(x, z)
        x = self.rb3(x, z)
        x = self.rb4(x, z)
        x = self.rb5(x, z)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, x.size(1))
        return x

def get_network(n_channels, ndf):
    return ResnetBase(input_nc=n_channels,
                      z_dim=3,
                      nf=ndf,
                      n_out=100)

if __name__ == '__main__':
    net = get_network(3, 64)
    xfake = torch.randn((4, 3, 64, 64))
    zfake = torch.randn((4, 3))
    print(net(xfake, zfake).shape)
