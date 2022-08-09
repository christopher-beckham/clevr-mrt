import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .shared import networks
from . import discriminators
from functools import partial

from .shared.networks import ConvBlock3d

class HoloEncoderMimick(nn.Module):
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
        super(HoloEncoderMimick, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        self.encoder_conv = nn.ModuleList([])
        self.encoder_norm = nn.ModuleList([])

        # Previously, n_ds=5 for 64px images
        # and n_ds=3 for 32px images.
        print("img siuze", im_size)
        n_downsampling  = 3 if im_size == 128 else 4
        #n_downsampling = int( (np.log(im_size) - np.log(16)) / np.log(2) )

        # im_size / 2**x = spat_size
        # log im_size - log(2**x) = log spat_size
        # log im_size - log spat_size = log(2**x)
        # x = (log im_size - log spat_size) / log(2)

        cc_dim = 2

        # Input preprocessor
        self.pre = nn.Sequential(
            nn.Conv2d(input_nc+cc_dim, ngf, kernel_size=7, padding=3,
                      bias=True),
            norm_layer_2d(ngf),
            nn.ReLU(True)
        )
        self.relu = nn.ReLU(True)

        for i in range(n_downsampling):
            mult = 2**i
            in_nf = ngf * mult
            out_nf = min(ngf * mult * 2, 1024)

            self.encoder_conv.append(
                nn.Conv2d(in_nf+cc_dim,
                          (out_nf),
                          kernel_size=3,
                          stride=2, padding=1, bias=True)
                )
            self.encoder_norm.append(
                norm_layer_2d(out_nf)
            )


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
        coords = self.coord_map(shape=input.size(-1))
        coords = coords.repeat(input.size(0), 1, 1, 1)
        h = self.pre(torch.cat((input, coords), dim=1))
        for conv_layer, norm_layer in zip(
                self.encoder_conv, self.encoder_norm):
            coords = self.coord_map(shape=h.size(-1))
            coords = coords.repeat(h.size(0), 1, 1, 1)
            h = torch.cat((h, coords), dim=1)
            h = conv_layer(h)
            h = norm_layer(h)
            h = self.relu(h)
        # Apply last conv
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
    gen = HoloEncoderMimick(input_nc=n_channels,
                            output_nc=n_channels,
                            ngf=ngf,
                            norm_layer_2d=norm_layer_2d,
                            im_size=im_size)
    return {
        'gen': gen
    }
