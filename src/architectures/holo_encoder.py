import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .shared import networks
from . import discriminators
from functools import partial

from .shared.networks import ConvBlock3d

class HoloEncoder(nn.Module):
    """
    x -> [conv2d + cc2d]
      -> z
      -> to 3d (spatial dim `vs`)
      -> [conv3d + cc3d]
      -> h

    """
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 vs=16,
                 theta_dim=32,
                 n_ds_3d=[2, 2, 4],
                 enc_dim=None,
                 norm_layer=nn.InstanceNorm3d,
                 norm_layer_2d=nn.InstanceNorm2d,
                 im_size=32):
        """
        """
        super(HoloEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        self.encoder_conv = nn.ModuleList([])
        self.encoder_norm = nn.ModuleList([])

        # Previously, n_ds=5 for 64px images
        # and n_ds=3 for 32px images.
        n_downsampling = int((np.log(im_size) - np.log(4)) / np.log(2))

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

        # Take the camera coordinates and
        # map them to a rotation matrix.
        self.cam_encode = nn.Sequential(
            nn.Linear(6, theta_dim),
            nn.BatchNorm1d(theta_dim),
            nn.ReLU(),
            nn.Linear(theta_dim, 6)
        )

        self.cam_infer = nn.Sequential(
            nn.Linear(6*2, theta_dim),
            nn.BatchNorm1d(theta_dim),
            nn.ReLU(),
            nn.Linear(theta_dim, 6)
        )

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
            if i != n_downsampling-1:
                self.encoder_norm.append(
                    norm_layer_2d(out_nf)
                )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.vs = vs
        self.final_nf = out_nf

        self.conv3d = nn.ModuleList([])

        # How many 3d convolutions should we do?
        # And how much do we downsample n filters?
        n_ds_3d = [1] + n_ds_3d
        for k in range(len(n_ds_3d)-1):
            self.conv3d.append(ConvBlock3d(out_nf // n_ds_3d[k] + 3,
                                           out_nf // n_ds_3d[k+1],
                                           norm_layer))

    def cam_infer(self, cam1, cam2):
        return self.cam_infer(
            torch.cat((cam1, cam2), dim=1)
        )

    def _split(self, z):
        len_ = z.size(1)
        mean = z[:, 0:(len_//2)]
        var = z[:, (len_//2):]
        return mean, var

    def _rshp2d(self, z):
        return z.view(-1, z.size(1), 1, 1)

    def _rshp3d(self, z):
        return z.view(-1, z.size(1), 1, 1, 1)

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
        z_coords = z_coord_row.unsqueeze(0).\
            view(-1, m, 1, 1).repeat(1, 1, n, o)
        return torch.cat([x_coords, y_coords, z_coords], 0)

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

    def enc2vol(self, enc):

        # Use spatial broadcast trick and tile.
        h = enc.view(enc.size(0), -1, 1, 1, 1).repeat(1, 1,
                                                      self.vs,
                                                      self.vs,
                                                      self.vs)
        # Add coord convs here.
        coords = self.coord_map_3d(self.vs)
        coords = coords.unsqueeze(0).repeat(h.size(0), 1, 1, 1, 1)

        # Do some non-strided 3d convolutions (+ 3d coord conv )
        # before HoloEncoder rotates it and passes it to the
        # probe.
        for conv3d_layer in self.conv3d:
            h = torch.cat((h, coords), dim=1)
            h = conv3d_layer(h,
                             None,
                             None)

        return h

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
        coords = self.coord_map(shape=h.size(-1))
        coords = coords.repeat(h.size(0), 1, 1, 1)
        h = torch.cat((h, coords), dim=1)
        h = self.encoder_conv[-1](h)
        # (n,f,h,w)
        h_flat = self.pool(h)
        h_flat = h_flat.view(-1, h_flat.size(1))
        return h_flat

def get_network(n_channels,
                ngf,
                n_ds_3d,
                theta_dim=32,
                use_bn=False):
    if use_bn:
        norm_layer = partial(nn.BatchNorm3d, affine=True)
        norm_layer_2d = partial(nn.BatchNorm2d, affine=True)
    else:
        norm_layer = partial(nn.InstanceNorm3d, affine=True)
        norm_layer_2d = partial(nn.InstanceNorm2d, affine=True)
    gen = HoloEncoder(input_nc=n_channels,
                      output_nc=n_channels,
                      ngf=ngf,
                      n_ds_3d=n_ds_3d,
                      vs=16,
                      theta_dim=theta_dim,
                      norm_layer=norm_layer,
                      norm_layer_2d=norm_layer_2d,
                      im_size=128)
    return {
        'gen': gen
    }
