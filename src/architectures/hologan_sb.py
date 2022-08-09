import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .shared import networks
from . import discriminators
from functools import partial

from .shared.networks import (ConvBlock3d,
                              ConvBlockUpsample2d)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class HoloGanSpatialBroadcast(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 nvf=64,
                 ndf=64,
                 vs=4,
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
        super(HoloGanSpatialBroadcast, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ndf = ndf
        self.gpu_ids = gpu_ids
        self.vs = vs
        self.learn_template = learn_template

        self.relu = nn.ReLU()

        self.conv3d = nn.ModuleList([])
        self.conv3d_post = nn.ModuleList([])
        if old_affine:
            # BUG where affine=False for 3d resblocks.
            conv3d_norm_layer = partial(norm_layer, affine=False)
        else:
            conv3d_norm_layer = norm_layer

        for k in range(len(v_ds)):
            if k == 0:
                # Has to be resblock3d specifically,
                # since it is strided
                n_in = (nvf+3) if not self.learn_template else (nvf*2)+3
                self.conv3d.append(ConvBlock3d(n_in,
                                               int(nvf//v_ds[0]),
                                               conv3d_norm_layer))
            elif k == 1:
                # Similar case to above.
                self.conv3d.append(ConvBlock3d(int(nvf//v_ds[0])+3,
                                               int(nvf//v_ds[1]),
                                               conv3d_norm_layer))
            else:
                self.conv3d.append(ConvBlock3d(int(nvf//v_ds[k-1])+3,
                                               int(nvf//v_ds[k]),
                                               conv3d_norm_layer))
        # For after rotation is done
        for k in range(n_postproc_rot):
            self.conv3d_post.append(
                ConvBlock3d(int(nvf//v_ds[-1]),
                            int(nvf//v_ds[-1]),
                            conv3d_norm_layer)
            )

        # e.g. 256*2 * (4*2) = 512*8 = 4096

        decoder = []
        if not pad_rot:
            ngf_new = int(nvf//v_ds[-1]) * (vs*(4))
            n_decodes = int(np.log(im_size / (vs*4)) / np.log(2))
        else:
            ngf_new = int(nvf//v_ds[-1]) * (vs*(8))
            n_decodes = int(np.log(im_size / (vs*4*2)) / np.log(2))

        self.slim = nn.Conv2d(ngf_new,
                              ngf_new // slim_factor,
                              kernel_size=1)

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


        #self.decoder = nn.Sequential(*decoder)
        self.post = nn.Sequential(*post)

        self.dec = nn.Sequential(*decoder)

        self.q = nn.Sequential(
            Flatten(),
            nn.Linear(ndf*8*4*4, nvf)
        )

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

def get_network(n_channels,
                nvf,
                ndf,
                vs=4,
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

    gen = HoloGanSpatialBroadcast(input_nc=n_channels,
                                  output_nc=3,
                                  nvf=nvf,
                                  ndf=ndf,
                                  vs=vs,
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
                                          n_classes=nvf,
                                          sigmoid=True,
                                          spec_norm=True)

    disc_enc = None

    return {
        'gen': gen,
        'disc_x': disc_x,
        'disc_enc': disc_enc
    }
