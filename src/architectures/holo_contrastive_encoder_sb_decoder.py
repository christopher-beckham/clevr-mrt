import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .shared import networks
from . import discriminators
from functools import partial

from .shared.networks import (ConvBlock3d,
                              Conv3d,
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
        theta = [nn.Conv2d(input_nc*2,
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
                      z_dim,
                      norm_layer_2d,
                      n_downsampling):
        pre = nn.Sequential(
            nn.Conv2d((input_nc*2)+2, ngf, kernel_size=7, padding=3,
                      bias=True),
            norm_layer_2d(ngf),
            nn.ReLU(True)
        )

        encoder_conv = nn.ModuleList([])
        encoder_norm = nn.ModuleList([])

        for i in range(n_downsampling):
            mult = 2**i
            in_nf = ngf * mult
            if i == n_downsampling-1:
                out_nf = z_dim
            else:
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
                 n_postproc_rot=2,
                 norm_layer=nn.InstanceNorm3d,
                 norm_layer_2d=nn.InstanceNorm2d,
                 im_size=32):

        super(HoloContrastiveEncoder, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.vs = vs

        self.relu = nn.ReLU()

        # How many downsamples to get from im_size to 4x4px?
        n_downsampling = int( (np.log(im_size) - np.log(4)) / np.log(2) )
        self.n_ds = n_downsampling

        self.theta = self._make_angle_encoder(input_nc,
                                              ngf,
                                              norm_layer_2d,
                                              n_downsampling)

        self.pre, self.encoder_conv, self.encoder_norm = \
            self._make_encoder(input_nc,
                               ngf,
                               z_dim,
                               norm_layer_2d,
                               n_downsampling)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_3d = nn.AdaptiveAvgPool3d(1)

        self.relu = nn.ReLU()

        if n_postproc_rot > 0:
            postproc = []
            for j in range(n_postproc_rot):
                postproc.append(
                    ConvBlock3d(z_dim, z_dim, norm_layer, stride=1)
                )
            self.postproc = nn.Sequential(*postproc)
        else:
            self.postproc = nn.Identity()

        # Define the module which maps from z to h.
        self.ztoh = nn.ModuleList([])
        v_ds_ztoh = list(v_ds) + [v_ds[-1]]
        for k in range(len(v_ds_ztoh)):
            if k == 0:
                self.ztoh.append(ConvBlock3d(z_dim+3,
                                             int(z_dim//v_ds_ztoh[0]),
                                             norm_layer))
            elif k == len(v_ds_ztoh)-1:
                # conv blocks have relus at the end, so make the
                # last 'block' a single 3d conv
                self.ztoh.append(Conv3d(
                    int(z_dim//v_ds_ztoh[k]),
                    int(z_dim//v_ds_ztoh[k])))
            else:
                self.ztoh.append(ConvBlock3d(int(z_dim//v_ds_ztoh[k-1])+3,
                                             int(z_dim//v_ds_ztoh[k]),
                                             norm_layer))

        # Define the module which maps from h to z.
        self.htoz = nn.ModuleList([])
        v_ds_htoz = list(v_ds_ztoh)[::-1]
        for k in range(1, len(v_ds_htoz)+1):
            if k == len(v_ds_htoz):
                self.htoz.append(Conv3d(z_dim, z_dim, stride=2))
            else:
                self.htoz.append(ConvBlock3d(int(z_dim//v_ds_htoz[k-1])+3,
                                             int(z_dim//v_ds_htoz[k]),
                                             norm_layer,
                                             stride=2))

    def forward(self, input):
        enc, h = self.encoder(input)
        dec = self.decoder(enc, h)
        return dec

    def enc2vol(self, enc):
        h = enc.view(enc.size(0), -1, 1, 1, 1).repeat(1, 1,
                                                      self.vs*4,
                                                      self.vs*4,
                                                      self.vs*4)
        # Add coord convs here
        coords = self.coord_map_3d(self.vs*4)
        coords = coords.unsqueeze(0).repeat(h.size(0), 1, 1, 1, 1)

        for i, conv3d_layer in enumerate(self.ztoh):
            #if i == 0 and self.learn_template:
            #    vol = self.vol.unsqueeze(0).\
            #        repeat(h.size(0), 1, 1, 1, 1)
            #    h = torch.cat((h, vol), dim=1)
            if i != len(self.ztoh)-1:
                h = torch.cat((h, coords), dim=1)
            h = conv3d_layer(h,
                             None,
                             None)

        return h

    def vol2enc(self, h, debug=False):
        for i, conv3d_layer in enumerate(self.htoz):
            coords = self.coord_map_3d( (self.vs*4) // 2**(i) )
            coords = coords.unsqueeze(0).repeat(h.size(0), 1, 1, 1, 1)
            if i != len(self.htoz)-1:
                h = torch.cat((h, coords), dim=1)
            h = conv3d_layer(h,
                             None,
                             None)


        h = self.pool_3d(h)
        h = h.view(-1, h.size(1))

        if debug:
            print("vol2enc:", h.shape)
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

    def encode(self, input1, input2=None, debug=False):
        if input2 is None:
            input = torch.cat((input1, input1), dim=1)
        else:
            input = torch.cat((input1, input2), dim=1)
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

        if debug:
            print("z shape:")
            print(z.shape)

        z = self.pool(z)
        if debug:
            print("after pooling z:")
            print(z.shape)

        z = z.view(-1, z.size(1))

        # Also extract the predicted angle.
        theta = self.theta(input)
        theta = theta.view(-1, theta.size(1))

        return z, theta

def get_network(n_channels,
                ngf,
                ndf,
                vs=16,
                z_dim=64,
                v_ds=[2, 2, 2, 4, 4],
                n_postproc_rot=2,
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
                                 z_dim=z_dim,
                                 n_postproc_rot=n_postproc_rot,
                                 norm_layer=norm_layer,
                                 norm_layer_2d=norm_layer_2d,
                                 im_size=im_size)

    return {
        'gen': gen,
        'disc': None
    }

if __name__ == '__main__':

    gen = get_network(
        n_channels=3,
        ngf=8,
        ndf=0,
        vs=16,
        v_ds=[1,2,4],
        n_postproc_rot=0,
        im_size=256,
        use_bn=True
    )['gen']
    gen = gen.cuda()

    print(gen)
    print("Test encoding...")
    xfake = torch.randn((4,3,256,256)).cuda()
    enc = gen.encode(xfake, debug=True)[0]

    h = gen.enc2vol(enc)
    print("h", h.shape)
    enc_again = gen.vol2enc(h)

    print(enc_again.shape)
