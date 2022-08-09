import torch
from torch import nn
import numpy as np

class ResnetBlock3d(nn.Module):
    def __init__(self, dim, norm_layer, stride=1):
        super(ResnetBlock3d, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, stride=1)

    def build_conv_block(self, dim, norm_layer, stride=1):
        conv_block = []
        p = 1

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=True),
                       norm_layer(dim),
                       nn.ReLU(True)]

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=True),
                       norm_layer(dim)]
        #bypass = []
        #if stride > 1:
        #    conv_block += [nn.AvgPool2d(2, stride=stride, padding=0)]
        #    bypass += [nn.AvgPool2d(2, stride=stride, padding=0)]
        #self.bypass = nn.Sequential(*bypass)

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetBlock2d(nn.Module):
    def __init__(self, dim, norm_layer, stride=1):
        super(ResnetBlock2d, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, stride=1)

    def build_conv_block(self, dim, norm_layer, stride=1):
        conv_block = []
        p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True),
                       norm_layer(dim),
                       nn.ReLU(True)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def make_encoder(ngf, n_downsampling, norm_layer=nn.BatchNorm3d):
    encoder = []
    for i in range(n_downsampling):
        mult = 2**i
        encoder += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                              stride=2, padding=1, bias=True),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)]
        encoder += [ResnetBlock3d(ngf*mult*2, norm_layer)]
    return nn.Sequential(*encoder)

def make_decoder(ngf, n_downsampling, norm_layer=nn.BatchNorm3d):
    decoder = []
    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        decoder += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1,
                                       bias=True),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
        decoder += [ResnetBlock3d(int(ngf * mult / 2),
                                norm_layer=norm_layer)
                   ]
    return nn.Sequential(*decoder)


def make_decoder_2d(ngf, n_downsampling, norm_layer=nn.BatchNorm2d):
    decoder = []
    for i in range(n_downsampling):
        mult = 2**i
        decoder += [nn.ConvTranspose2d(ngf // mult, ngf // (mult*2),
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1,
                                       bias=True),
                    norm_layer(ngf // (mult*2)),
                    nn.ReLU(True)]
        decoder += [ResnetBlock2d(ngf // (mult*2),
                                norm_layer=norm_layer)
                   ]
    return nn.Sequential(*decoder)
