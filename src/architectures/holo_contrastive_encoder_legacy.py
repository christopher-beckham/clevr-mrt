import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .shared import networks
from functools import partial

from .shared.networks import (ConvBlock3d,
                              ConvBlockUpsample2d,
                              ResBlock3d)
from .holo_encoder_base import HoloEncoderBase


class HoloContrastiveEncoder(HoloEncoderBase):

    def _make_encoder(self,
                      input_nc,
                      ngf,
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
                 vs=16,
                 v_ds=[4, 8],
                 n_postproc_rot=0,
                 pool_mode='avg',
                 norm_layer=nn.InstanceNorm3d,
                 norm_layer_2d=nn.InstanceNorm2d,
                 im_size=256):
        """
        vs: what should the spatial dimension of the latent volume
          be?
        v_ds: supposed to be used for vol2enc but due to a bug
          it has no effect here. So in this case, vol2enc simply
          performs a avgpool operation on the volume and enc
          is simply the averaged feature maps.
        n_postproc_rot: initially was not used for contrastive
          training step, but now is the postprocessor in the
          second stage (FILM).
        """

        super(HoloContrastiveEncoder, self).__init__()

        if pool_mode not in ['avg', 'max']:
            raise Exception("`pool_mode` must be either avg or max")

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.vs = vs

        self.relu = nn.ReLU()

        # Take the camera coordinates and
        # map them to a rotation matrix.

        # How many downsamples to get from im_size to vs?
        n_downsampling = int( (np.log(im_size) - np.log(vs)) / np.log(2) )
        self.n_ds = n_downsampling

        self.pre, self.encoder_conv, self.encoder_norm = \
            self._make_encoder(input_nc,
                               ngf,
                               norm_layer_2d,
                               n_downsampling)

        if pool_mode == 'avg':
            self.pool = nn.AdaptiveAvgPool3d(1)
        else:
            self.pool = nn.AdaptiveMaxPool3d(1)

        self.relu = nn.ReLU()

        nvf = (ngf*(2**n_downsampling)) // vs
        self.nvf = nvf

        self.conv3d = nn.ModuleList([])
        self.conv3d_post = nn.ModuleList([])
        # ** BUG ** this module is never used. It
        # was meant to be adding to self.conv3d_post,
        # not self.conv3d, but self.conv3d isn't
        # used either so this is completely useless.
        for k in range(len(v_ds)):
            if k == 0:
                # Has to be resblock3d specifically,
                # since it is strided
                self.conv3d.append(ConvBlock3d(nvf+3,
                                               int(nvf//v_ds[0]),
                                               norm_layer,
                                               stride=2))
            elif k == 1:
                # Similar case to above.
                self.conv3d.append(ConvBlock3d(int(nvf//v_ds[0])+3,
                                               int(nvf//v_ds[1]),
                                               norm_layer,
                                               stride=2))
            else:
                self.conv3d.append(ConvBlock3d(int(nvf//v_ds[k-1])+3,
                                               int(nvf//v_ds[k]),
                                               norm_layer,
                                               stride=2))
        # ONLY USED FOR FILM STAGE
        # This is analogous to the postprocessor
        # in holo_encoder_3dlite_pt.py in architectures.
        postproc = []
        for k in range(n_postproc_rot):
            postproc.append(ConvBlock3d(nvf*(2**k),
                                        nvf*(2**(k+1)),
                                        norm_layer))
        self.postproc = nn.Sequential(*postproc)

    @property
    def postprocessor(self):
        return self.postproc

    def postprocess(self, x):
        return self.postproc(x)

    #def encode(self, input1, input2=None):
    #    if self.multi_gpu:
    #        nn.parallel.data_parallel(
    #            module=self._encode,
    #            inputs=[input1, input2]
    #        )
    #    else:
    #        return self._encode(input1, input2)

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

        # (bs, 1024, 16, 16)
        # (bs, 1024//16, 16, 16, 16)
        spat_dim = z.size(-1)
        z = z.view(-1,
                   z.size(1)//spat_dim,
                   spat_dim,
                   spat_dim,
                   spat_dim)

        return z, None

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


def get_network(n_channels,
                ngf,
                ndf,
                vs=16,
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
                                 n_postproc_rot=n_postproc_rot,
                                 norm_layer=norm_layer,
                                 norm_layer_2d=norm_layer_2d,
                                 im_size=im_size)

    return {
        'gen': gen,
        'disc': None
    }
