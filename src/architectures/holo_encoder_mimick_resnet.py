import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .shared import networks
from . import discriminators
from functools import partial

from .shared.networks import ConvBlock3d

from .clevr.layers import ResidualBlock

class HoloEncoderMimickResnet(nn.Module):
    """

    """
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer_2d=nn.InstanceNorm2d,
                 im_size=32):
        """
        """
        super(HoloEncoderMimickResnet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        self.convs = nn.ModuleList([])

        # Previously, n_ds=5 for 64px images
        # and n_ds=3 for 32px images.

        n_downsampling  = 3 if im_size == 128 else 4

       # Input preprocessor
        self.pre = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3,
                      bias=True),
            norm_layer_2d(ngf),
            nn.ReLU(True)
        )

        for i in range(n_downsampling):
            mult = 2**i
            in_nf = ngf * mult
            out_nf = min(ngf * mult * 2, 1024)

            block = ResidualBlock(in_dim=in_nf,
                                  out_dim=out_nf,
                                  with_batchnorm=True,
                                  downsample=True)
            self.convs.append(block)


    def coord_map(self, shape, start=-1, end=1):
        """
        Gives, a 2d shape tuple, returns two mxn coordinate maps,
        Ranging min-max in the x and y directions, respectively.
        """
        m = n = shape
        x_coord_row = torch.linspace(start, end, steps=n).\
            type(torch.cuda.FloatTensor)
        y_coord_row = torch.linspace(start, end, steps=m).\
            type(torch.cuda.FloatTensor)
        x_coords = x_coord_row.unsqueeze(0).\
            expand(torch.Size((m, n))).unsqueeze(0)
        y_coords = y_coord_row.unsqueeze(1).\
            expand(torch.Size((m, n))).unsqueeze(0)
        return torch.cat([x_coords, y_coords], 0)

    def forward(self, input):
        #enc, h = self.encoder(input)
        #dec = self.decoder(enc, h)
        #return dec
        return None

    def enc2vol(self, x):
        return x

    def encode(self, input):
        h = self.pre(input)
        for block in self.convs:
            h = block(h)
        return h

def get_network(n_channels,
                ngf,
                theta_dim=32,
                im_size=224,
                use_bn=False):
    if use_bn:
        norm_layer_2d = partial(nn.BatchNorm2d, affine=True)
    else:
        norm_layer_2d = partial(nn.InstanceNorm2d, affine=True)
    gen = HoloEncoderMimickResnet(input_nc=n_channels,
                                  output_nc=n_channels,
                                  ngf=ngf,
                                  norm_layer_2d=norm_layer_2d,
                                  im_size=im_size)
    return {
        'gen': gen
    }
