import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .shared import networks
from . import discriminators
from functools import partial

from .shared.networks import (ConvBlock3d,
                              ConvBlockUpsample2d,
                              ResBlock3d)
from .holo_encoder_base import HoloEncoderBase

#from .discriminators_3d import Discriminator


class HoloContrastiveEncoder(HoloEncoderBase):

    def _make_angle_encoder(self,
                            input_nc,
                            ngf,
                            norm_layer_2d,
                            n_downsampling):
        theta_dim = 7+1
        div_factor = 2
        theta = [nn.Conv2d(input_nc*1,
                           ngf // div_factor,
                           kernel_size=7,
                           padding=3,
                           bias=True),
                 norm_layer_2d(ngf // div_factor),
                 nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2**i
            in_nf = ngf * mult // div_factor
            if i==(n_downsampling-1):
                out_nf = theta_dim
            else:
                out_nf = ngf * mult * 2 // div_factor
            theta += [nn.Conv2d(in_nf,
                                out_nf,
                                kernel_size=3,
                                stride=2, padding=1, bias=True)]
            if i != n_downsampling-1:
                theta += [norm_layer_2d((ngf * mult * 2 // div_factor)),
                          nn.ReLU(True)]
        theta += [nn.AdaptiveAvgPool2d(1)]
        theta = nn.Sequential(*theta)
        return theta

    def _make_encoder(self,
                      input_nc,
                      ngf,
                      norm_layer_2d,
                      n_downsampling):
        pre = nn.Sequential(
            nn.Conv2d((input_nc*1)+2, ngf, kernel_size=7, padding=3,
                      bias=True),
            norm_layer_2d(ngf),
            nn.ReLU(True)
        )

        encoder_conv = nn.ModuleList([])
        encoder_norm = nn.ModuleList([])

        for i in range(n_downsampling):
            mult = 2**i
            in_nf = ngf * mult
            out_nf = ngf * mult * 2

            encoder_conv.append(
                nn.Conv2d(in_nf+2,
                          (out_nf),
                          kernel_size=3,
                          stride=2, padding=1, bias=True)
            )

            if i != n_downsampling-1:
                encoder_norm.append(norm_layer_2d(ngf*mult*2))
        return pre, encoder_conv, encoder_norm


    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 z_dim=64,
                 vs=16,
                 v_ds=[4, 8],
                 n_postproc=2,
                 norm_layer=nn.InstanceNorm3d,
                 norm_layer_2d=nn.InstanceNorm2d,
                 im_size=32):

        super(HoloContrastiveEncoder, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.vs = vs

        self.relu = nn.ReLU()

        # Take the camera coordinates and
        # map them to a rotation matrix.

        # NOTE: legacy, this has now been
        # moved to the probe class.
        theta_dim=32
        self.cam_encode = nn.Sequential(
            nn.Linear(6, theta_dim),
            nn.BatchNorm1d(theta_dim),
            nn.ReLU(),
            nn.Linear(theta_dim, 6)
        )

        # How many downsamples to get from im_size to vs?
        n_downsampling = int( (np.log(im_size) - np.log(vs)) / np.log(2) )

        self.n_ds = n_downsampling

        self.theta = self._make_angle_encoder(input_nc,
                                              ngf,
                                              norm_layer_2d,
                                              n_downsampling)

        self.pre, self.encoder_conv, self.encoder_norm = \
            self._make_encoder(input_nc,
                               ngf,
                               norm_layer_2d,
                               n_downsampling)

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.relu = nn.ReLU()

        nvf = (ngf*(2**n_downsampling)) // vs
        if n_postproc > 0:
            postproc = []
            for j in range(n_postproc):
                postproc.append(
                    ConvBlock3d(nvf*(2**j),
                                nvf*(2**(j+1)),
                                norm_layer,
                                stride=1)
                )
            self.postproc = nn.Sequential(*postproc)
        else:
            self.postproc = nn.Identity()

        nvf2 = nvf*(2**(j+1))
        self.conv3d_post = nn.ModuleList([])
        for k in range(len(v_ds)):
            if k == 0:
                # Has to be resblock3d specifically,
                # since it is strided
                self.conv3d_post.append(ConvBlock3d(nvf2+3,
                                                    int(nvf2//v_ds[0]),
                                                    norm_layer,
                                                    stride=2))
            elif k == 1:
                # Similar case to above.
                self.conv3d_post.append(ConvBlock3d(int(nvf2//v_ds[0])+3,
                                                    int(nvf2//v_ds[1]),
                                                    norm_layer,
                                                    stride=2))
            else:
                self.conv3d_post.append(ConvBlock3d(int(nvf2//v_ds[k-1])+3,
                                                    int(nvf2//v_ds[k]),
                                                    norm_layer,
                                                    stride=2))

    def forward(self, input):
        enc, h = self.encoder(input)
        dec = self.decoder(enc, h)
        return dec

    def vol2enc(self, h):

        # Add coord convs here
        for conv3d_layer in self.conv3d_post:
            coords = self.coord_map_3d(h.size(-1))
            coords = coords.unsqueeze(0).repeat(h.size(0), 1, 1, 1, 1)
            h = torch.cat((h, coords), dim=1)
            h = conv3d_layer(h,
                             None,
                             None)
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        return h

    def post_rot(self, h):
        #for conv3d_layer in self.conv3d_post:
        #    h = conv3d_layer(h,
        #                     None,
        #                     None)
        #return h
        return h

    def coord_map_3d(self, shape, start=-1, end=1):
        m = n = o = shape
        x_coord_row = torch.linspace(start, end, steps=n).\
            type(torch.cuda.FloatTensor)
        y_coord_row = torch.linspace(start, end, steps=m).\
            type(torch.cuda.FloatTensor)
        z_coord_row = torch.linspace(start, end, steps=o).\
            type(torch.cuda.FloatTensor)

        x_coords = x_coord_row.unsqueeze(0).\
            expand(torch.Size((m, n, o))).unsqueeze(0)
        y_coords = y_coord_row.unsqueeze(1).\
            expand(torch.Size((m, n, o))).unsqueeze(0)
        #z_coords = z_coord_row.unsqueeze(2).expand(torch.Size((m, n, o))).unsqueeze(0)
        z_coords = z_coord_row.unsqueeze(0).\
            view(-1, m, 1, 1).repeat(1, 1, n, o)
        return torch.cat([x_coords, y_coords, z_coords], 0)

    def coord_map(self, shape, start=-1, end=1):
        """
        Gives, a 2d shape tuple, returns two mxn coordinate maps,
        Ranging min-max in the x and y directions, respectively.
        """
        m = shape
        n = shape
        x_coord_row = torch.linspace(start, end, steps=n).\
            type(torch.cuda.FloatTensor)
        y_coord_row = torch.linspace(start, end, steps=m).\
            type(torch.cuda.FloatTensor)
        x_coords = x_coord_row.unsqueeze(0).\
            expand(torch.Size((m, n))).unsqueeze(0)
        y_coords = y_coord_row.unsqueeze(1).\
            expand(torch.Size((m, n))).unsqueeze(0)
        return torch.cat([x_coords, y_coords], 0)

    def encode(self, input1, input2=None):
        input = input1
        # Encode the image into two halves:
        # the deterministic component, and
        # the stochastic one.
        coords = self.coord_map(shape=input.size(-1))
        coords = coords.repeat(input.size(0), 1, 1, 1)
        z = self.pre(torch.cat((input, coords), dim=1))
        for conv_layer, norm_layer in zip(
                self.encoder_conv, self.encoder_norm):
            coords = self.coord_map(shape=z.size(-1))
            coords = coords.repeat(z.size(0), 1, 1, 1)
            z = torch.cat((z, coords), dim=1)
            z = conv_layer(z)
            z = norm_layer(z)
            z = self.relu(z)
        # Apply last conv
        coords = self.coord_map(shape=z.size(-1))
        coords = coords.repeat(z.size(0), 1, 1, 1)
        z = torch.cat((z, coords), dim=1)
        z = self.encoder_conv[-1](z)

        # (bs, 1024, 16, 16)
        # (bs, 1024//16, 16, 16, 16)
        spat_dim = z.size(-1)
        z = z.view(-1,
                   z.size(1)//spat_dim,
                   spat_dim,
                   spat_dim,
                   spat_dim)

        z = self.postproc(z)

        #z = self.pool(z)

        #enc_ = z.view(-1, z.size(1))

        # Also extract the predicted angle.
        theta = self.theta(input)
        theta = theta.view(-1, theta.size(1))

        return z, theta

def get_network(n_channels,
                ngf,
                ndf,
                vs=16,
                v_ds=[2, 2, 2, 4, 4],
                n_postproc=2,
                im_size=128,
                use_bn=True):
    if use_bn:
        norm_layer = partial(nn.BatchNorm3d, affine=True)
        norm_layer_2d = partial(nn.BatchNorm2d, affine=True)
    else:
        norm_layer = partial(nn.InstanceNorm3d, affine=True)
        norm_layer_2d = partial(nn.InstanceNorm2d, affine=True)

    gen = HoloContrastiveEncoder(input_nc=n_channels,
                                 output_nc=3,
                                 ngf=ngf,
                                 vs=vs,
                                 v_ds=v_ds,
                                 n_postproc=n_postproc,
                                 norm_layer=norm_layer,
                                 norm_layer_2d=norm_layer_2d,
                                 im_size=im_size)

    return {
        'gen': gen,
        'disc': None
    }
