import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .shared import networks
from . import discriminators
from functools import partial

from .shared.networks import (ConvBlock3d,
                              ConvBlockUpsample2d)

class HoloVaeSpatialBroadcast(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 nvf=64,
                 vs=4,
                 n_ds=5,
                 v_ds=[4, 8],
                 n_postproc_rot=2,
                 learn_template=False,
                 norm_layer=nn.InstanceNorm3d,
                 norm_layer_2d=nn.InstanceNorm2d,
                 im_size=32,
                 slim_factor=2,
                 pad_rot=False,
                 old_affine=True,
                 gpu_ids=[]):
        super(HoloVaeSpatialBroadcast, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.vs = vs
        self.learn_template = learn_template

        # Previously, n_ds=5 for 64px images
        # and n_ds=3 for 32px images.
        n_downsampling = n_ds

        ###########################################################

        theta_dim = 7*2
        div_factor = 2
        theta = [nn.Conv2d(input_nc*2, ngf // div_factor, kernel_size=7, padding=3,
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

        ###########################################################

        self.pre = nn.Sequential(
            nn.Conv2d((input_nc*2)+2, ngf, kernel_size=7, padding=3,
                      bias=True),
            norm_layer_2d(ngf),
            nn.ReLU(True)
        )

        self.encoder_conv = nn.ModuleList([])
        self.encoder_norm = nn.ModuleList([])
        self.relu = nn.ReLU()

        for i in range(n_downsampling):
            mult = 2**i
            in_nf = ngf * mult
            if i==(n_downsampling-1):
                # `enc_dim` allows us to override the final
                # number of feature maps, otherwise it will
                # simply be ngf*(2**downsampling)`
                out_nf = nvf
            else:
                out_nf = ngf * mult * 2

            self.encoder_conv.append(
                nn.Sequential(
                    nn.Conv2d(in_nf+2,
                              (out_nf),
                              kernel_size=3,
                              stride=2, padding=1, bias=True)
                )
            )

            if i != n_downsampling-1:
                self.encoder_norm.append(norm_layer_2d(ngf*mult*2))

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.ReLU()

        self.conv3d = nn.ModuleList([])
        self.conv3d_post = nn.ModuleList([])
        if old_affine:
            # BUG where affine=False for 3d resblocks.
            conv3d_norm_layer = partial(norm_layer, affine=False)
        else:
            conv3d_norm_layer = norm_layer

        if self.learn_template:
            vol = torch.ones((nvf, vs*4, vs*4, vs*4)) # 4x4x4
            torch.nn.init.xavier_uniform_(vol)
            self.vol = nn.Parameter(vol, requires_grad=True)

        for k in range(len(v_ds)):
            if k == 0:
                # Has to be resblock3d specifically,
                # since it is strided
                n_in = (nvf+3) if not self.learn_template else (nvf*2)+3
                self.conv3d.append(ConvBlock3d(n_in,
                                               nvf//v_ds[0],
                                               conv3d_norm_layer))
            elif k == 1:
                # Similar case to above.
                self.conv3d.append(ConvBlock3d(nvf//v_ds[0]+3,
                                               nvf//v_ds[1],
                                               conv3d_norm_layer))
            else:
                self.conv3d.append(ConvBlock3d(nvf//v_ds[k-1]+3,
                                               nvf//v_ds[k],
                                               conv3d_norm_layer))
        # For after rotation is done
        for k in range(n_postproc_rot):
            self.conv3d_post.append(
                ConvBlock3d(nvf//v_ds[-1],
                            nvf//v_ds[-1],
                            conv3d_norm_layer)
            )

        # e.g. 256*2 * (4*2) = 512*8 = 4096


        decoder = []
        if not pad_rot:
            ngf_new = (nvf//v_ds[-1]) * (vs*(4))
            n_decodes = int(np.log(im_size/ (vs*4)) / np.log(2))
        else:
            ngf_new = (nvf//v_ds[-1]) * (vs*(8))
            n_decodes = int(np.log(im_size/ (vs*4*2)) / np.log(2))

        self.slim = nn.Conv2d(ngf_new, ngf_new // slim_factor, kernel_size=1)

        ngf_new = ngf_new // slim_factor

        for i in range(n_decodes):
            j = 2**i
            jj = 2**(i+1)
            # Stride conv then proc conv
            decoder.append(ConvBlockUpsample2d(ngf_new // j,
                                               ngf_new // jj,
                                               norm_layer_2d))
            #decoder.append(ConvBlock2d(ngf_new // jj, ngf_new // jj))

        post = []
        post += [nn.ReflectionPad2d(3)]
        post += [nn.Conv2d(ngf_new // jj,
                           output_nc,
                           kernel_size=7,
                           padding=0)]
        post += [nn.Tanh()]


        self.theta = nn.Sequential(*theta)
        #self.decoder = nn.Sequential(*decoder)
        self.post = nn.Sequential(*post)

        self.dec = nn.Sequential(*decoder)

    def _split(self, z):
        len_ = z.size(1)
        mean = z[:, 0:(len_//2)]
        var = z[:, (len_//2):]
        return mean, var

    def _rshp2d(self, z):
        return z.view(-1, z.size(1), 1, 1)

    def _rshp3d(self, z):
        return z.view(-1, z.size(1), 1, 1, 1)

    def forward(self, input):
        enc, h = self.encoder(input)
        dec = self.decoder(enc, h)
        return dec

    def enc2vol(self, enc):
        #Z = self.vol.repeat((enc.size(0), 1, 1, 1, 1))

        #vs=4, so vs*4=16
        h = enc.view(enc.size(0), -1, 1, 1, 1).repeat(1, 1,
                                                      self.vs*4,
                                                      self.vs*4,
                                                      self.vs*4)
        # Add coord convs here
        coords = self.coord_map_3d(self.vs*4)
        coords = coords.unsqueeze(0).repeat(h.size(0), 1, 1, 1, 1)

        for i,conv3d_layer in enumerate(self.conv3d):
            if i == 0 and self.learn_template:
                vol = self.vol.unsqueeze(0).\
                    repeat(h.size(0), 1, 1, 1, 1)
                h = torch.cat((h, vol), dim=1)
            h = torch.cat((h, coords), dim=1)
            h = conv3d_layer(h,
                             None,
                             None)

        return h

    def post_rot(self, h):
        for conv3d_layer in self.conv3d_post:
            h = conv3d_layer(h,
                             None,
                             None)
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
        z = self.pool(z)

        enc_ = z.view(-1, z.size(1))

        # Also extract the predicted angle.
        theta = self.theta(input)
        theta = theta.view(-1, theta.size(1))

        return enc_, theta

    def decode(self, h):

        # Post-processing block here,
        # otherwise we see the zero padding
        # when we decode.
        #h = self.conv3d_post(h)

        h = h.contiguous().\
            view(-1, h.size(1)*h.size(2), h.size(3), h.size(4))
        # Ok, now decode in 2d but keep augmenting it with
        # the encoding.

        h = self.slim(h)

        h = self.dec(h)

        h = self.post(h)
        return h

from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

class ThetaDiscriminator(nn.Module):
    def __init__(self,
                 ndf,
                 theta_dim,
                 n_layers=4):
        super(ThetaDiscriminator, self).__init__()
        net = []
        for i in range(n_layers):
            if i == 0:
                n_in = theta_dim
            else:
                n_in = ndf
            if i != (n_layers)-1:
                layer = nn.Linear(n_in, ndf)
                nn.init.xavier_uniform(layer.weight.data, 1.)
                net += [
                    SpectralNorm(layer),
                    nn.ReLU()
                ]
            else:
                layer = nn.Linear(ndf, 1)
                nn.init.xavier_uniform(layer.weight.data, 1.)
                net += [SpectralNorm(layer)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return F.sigmoid(self.net(x))

def get_network(n_channels,
                ngf,
                nvf,
                ndf,
                vs=4,
                n_ds=4,
                v_ds=[2, 2, 2, 4, 4],
                n_postproc_rot=2,
                slim_factor=2,
                learn_template=False,
                im_size=128,
                use_bn=True,
                pad_rot=False,
                old_affine=True):
    if use_bn:
        norm_layer = partial(nn.BatchNorm3d, affine=True)
        norm_layer_2d = partial(nn.BatchNorm2d, affine=True)
    else:
        norm_layer = partial(nn.InstanceNorm3d, affine=True)
        norm_layer_2d = partial(nn.InstanceNorm2d, affine=True)

    gen = HoloVaeSpatialBroadcast(input_nc=n_channels,
                                  output_nc=3,
                                  ngf=ngf,
                                  nvf=nvf,
                                  vs=vs,
                                  n_ds=n_ds,
                                  v_ds=v_ds,
                                  n_postproc_rot=n_postproc_rot,
                                  learn_template=learn_template,
                                  norm_layer=norm_layer,
                                  norm_layer_2d=norm_layer_2d,
                                  im_size=im_size,
                                  slim_factor=slim_factor,
                                  pad_rot=pad_rot,
                                  old_affine=old_affine)
    disc_x = discriminators.Discriminator(nf=ndf,
                                          input_nc=n_channels,
                                          n_classes=0,
                                          sigmoid=False,
                                          spec_norm=True)
    #disc_enc = discriminators.DiscriminatorFC(
    #    n_in=enc_dim,
    #    n_out=1,
    #    nu=enc_dim*2,
    #    n_layers=2,
    #    spec_norm=True)

    disc_enc = ThetaDiscriminator(
        theta_dim=7,
        ndf=ndf
    )

    return {
        'gen': gen,
        'disc_x': disc_x,
        'disc_enc': disc_enc
    }
