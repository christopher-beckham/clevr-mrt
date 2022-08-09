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

class AvgMaxPoolConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
    def forward(self, x):
        z1 = self.avgpool(x)
        z1 = z1.view(z1.size(0), -1)
        z2 = self.maxpool(x)
        z2 = z2.view(z2.size(0), -1)
        return torch.cat((z1, z2), dim=1)

class HoloContrastiveEncoder(HoloEncoderBase):

    def _make_encoder(self,
                      input_nc,
                      ngf,
                      norm_layer_2d,
                      n_downsampling):

        pre = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3,
                      bias=True),
            norm_layer_2d(ngf),
            nn.ReLU(True)
        )

        layers = [pre]
        for i in range(n_downsampling):
            mult = 2**i
            in_nf = ngf * mult
            out_nf = ngf * mult * 2

            if i==(n_downsampling-1):
                # for y axis
                out_nf += 1
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_nf,
                                  (out_nf),
                                  kernel_size=3,
                                  stride=2, padding=1, bias=True)
                    )
                )

            else:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_nf,
                                  (out_nf),
                                  kernel_size=3,
                                  stride=2, padding=1, bias=True),
                        norm_layer_2d(out_nf),
                        nn.ReLU()
                    )
                )
        layers = nn.Sequential(*layers)

        return layers

    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 n_out=None,
                 vs=16,
                 v_ds=[4, 8],
                 pooling_mode='avg',
                 n_postproc_rot=0,
                 postproc_double_nf=True,
                 norm_layer=nn.InstanceNorm3d,
                 norm_layer_2d=nn.InstanceNorm2d,
                 extra_convs=False,
                 im_size=32):

        super(HoloContrastiveEncoder, self).__init__()

        # TODO: add option to disable coord conv in the encoder
        # to save memory

        if pooling_mode not in ['avg', 'max', 'none', 'avg_and_max']:
            raise Exception("pooling mode %s is not supported" % pooling_mode)

        print("DETECTED %i GPUs" % torch.cuda.device_count())

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

        self.encoder_layers = self._make_encoder(input_nc,
                                                 ngf,
                                                 norm_layer_2d,
                                                 n_downsampling)

        if pooling_mode == 'avg':
            self.pool = nn.AdaptiveAvgPool3d(1)
        elif pooling_mode == 'max':
            self.pool = nn.AdaptiveMaxPool3d(1)
        elif pooling_mode == 'avg_and_max':
            # Compute both an avg pool and a max pool
            # then concatenate their vectors together.
            self.pool = AvgMaxPoolConcat()
        else:
            # don't pool, just flatten
            self.pool = nn.Identity()

        self.pool_2d = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.ReLU()

        nvf = (ngf*(2**n_downsampling)) // vs
        self.nvf = nvf

        ##############################################
        # This section is the projection from h -> z #
        ##############################################
        self.conv3d_post = []
        for k in range(len(v_ds)):
            if k == 0:
                # Has to be resblock3d specifically,
                # since it is strided
                self.conv3d_post.append(ConvBlock3d(nvf,
                                                    int(nvf//v_ds[0]),
                                                    norm_layer,
                                                    stride=2))
            else:
                self.conv3d_post.append(ConvBlock3d(int(nvf//v_ds[k-1]),
                                                    int(nvf//v_ds[k]),
                                                    norm_layer,
                                                    stride=2))
        if len(v_ds) > 0:
            # We need an extra conv here since the resblocks'
            # outputs are relu'ed.
            pp = nn.Conv3d(int(nvf//v_ds[-1]),
                           int(nvf//v_ds[-1]),
                           kernel_size=3,
                           padding=1)
            self.conv3d_post.append(pp)
        self.conv3d_post += [self.pool]
        self.conv3d_post = nn.Sequential(*self.conv3d_post)
        ################################################

        # ONLY USED FOR FILM STAGE
        # This is analogous to the postprocessor
        # in holo_encoder_3dlite_pt.py in architectures.
        # Postprocessors will double the number of
        # feature maps at each block.
        postproc = []
        # e.g. if pp_mult == 2, then
        # we do nvf*2^2, nvf*2^4, etc.
        for k in range(n_postproc_rot):
            if postproc_double_nf:
                postproc.append(ConvBlock3d(nvf*(2**k),
                                            nvf*(2**(k+1)),
                                            norm_layer))
            else:
                postproc.append(ConvBlock3d(nvf,
                                            nvf,
                                            norm_layer))
        self.postproc = nn.Sequential(*postproc)

    @property
    def postprocessor(self):
        return self.postproc

    def postprocess(self, x):
        return self.postproc(x)

    def vol2enc(self, h):
        h = nn.parallel.data_parallel(
            self.conv3d_post,
            h
        )
        h = h.view(h.size(0), -1)
        return h

    def encode(self, input):

        # Encode the image into two halves:
        # the deterministic component, and
        # the stochastic one.
        z = nn.parallel.data_parallel(
            self.encoder_layers,
            input
        )

        z_ = z[:, 0:-1]
        q_ = z[:, -1:]
        q_ = self.pool_2d(q_).view(-1, 1)

        # (bs, 1024, 16, 16)
        # (bs, 1024//16, 16, 16, 16)
        spat_dim = z_.size(-1)
        # if spat_dim is a power of 2, then we're fine,
        # otherwise the depth dimension will be slightly
        # bigger.
        if not (np.log(spat_dim) / np.log(2)).is_integer():
            # Find the closest power of two to spat_dim,
            # e.g. if spat_dim==14 then answer is 16.
            depth_spat_dim = 2**int(np.ceil(np.log(spat_dim) / np.log(2)))
        else:
            depth_spat_dim = spat_dim

        z_ = z_.view(-1,
                     z_.size(1)//depth_spat_dim,
                     depth_spat_dim,
                     spat_dim,
                     spat_dim)

        return z_, q_

def get_network(n_channels,
                ngf,
                ndf,
                n_out=None,
                vs=16,
                v_ds=[2, 2, 2, 4, 4],
                pooling_mode='avg',
                n_postproc_rot=0,
                postproc_double_nf=True,
                im_size=128,
                extra_convs=False,
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
                                 n_out=n_out,
                                 vs=vs,
                                 v_ds=v_ds,
                                 pooling_mode=pooling_mode,
                                 n_postproc_rot=n_postproc_rot,
                                 postproc_double_nf=postproc_double_nf,
                                 norm_layer=norm_layer,
                                 norm_layer_2d=norm_layer_2d,
                                 extra_convs=extra_convs,
                                 im_size=im_size)

    return {
        'gen': gen,
        'disc': None
    }
