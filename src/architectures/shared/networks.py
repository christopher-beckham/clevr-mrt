"""
Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


--------------------------- LICENSE FOR pix2pix --------------------------------
BSD License

For pix2pix software
Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np

###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias, input_2d=False, stride=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_bias, input_2d, stride=1)

    def build_conv_block(self, dim, norm_layer, use_bias, input_2d, stride=1):
        if input_2d:
            conv_layer = nn.Conv2d
            pool_layer = nn.AvgPool2d
        else:
            conv_layer = nn.Conv3d
            pool_layer = nn.AvgPool3d
        conv_block = []
        p = 1
        conv_block += [conv_layer(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        conv_block += [conv_layer(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]
        bypass = []
        if stride > 1:
            conv_block += [pool_layer(2, stride=stride, padding=0)]
            bypass += [pool_layer(2, stride=stride, padding=0)]
        self.bypass = nn.Sequential(*bypass)

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.bypass(x) + self.conv_block(x)
        return out


class ResBlock3d(nn.Module):

    def __init__(self, in_ch, out_ch, norm_layer):
        super(ResBlock3d, self).__init__()

        self.conv1 = nn.ConvTranspose3d(in_ch,
                                        out_ch, 4, 2,
                                        padding=1,
                                        output_padding=0)
        self.bn = norm_layer(out_ch, affine=True)
        self.relu = nn.ReLU(True)

        nn.init.xavier_uniform(self.conv1.weight.data, 1.)

    def forward(self, inp, mean, sigma):
        if mean is None:
            mean = 0.
        if sigma is None:
            sigma = 1.
        x = self.conv1(inp)
        x = self.bn(x)*sigma + mean
        x = self.relu(x)
        return x

class ConvBlockUpsample2d(nn.Module):

    def __init__(self, in_ch, out_ch, norm_layer):
        super(ConvBlockUpsample2d, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_ch,
                                        out_ch, 4, 2,
                                        padding=1,
                                        output_padding=0)
        self.bn = norm_layer(out_ch, affine=True)
        self.relu = nn.ReLU(True)

        nn.init.xavier_uniform(self.conv1.weight.data, 1.)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBlock2d(nn.Module):

    def __init__(self, in_ch, out_ch, stride=2):
        super(ConvBlock2d, self).__init__()

        self.conv1 = nn.Conv2d(in_ch,
                               out_ch, 3, 1,
                               padding=1)
        self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        self.relu = nn.ReLU(True)

        nn.init.xavier_uniform(self.conv1.weight.data, 1.)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBlock3d(nn.Module):

    def __init__(self, in_ch, out_ch, norm_layer, stride=1):
        super(ConvBlock3d, self).__init__()

        self.conv1 = nn.Conv3d(in_ch,
                               out_ch,
                               3, stride=stride,
                               padding=1)
        self.bn = norm_layer(out_ch)
        self.relu = nn.ReLU(True)

        nn.init.xavier_uniform(self.conv1.weight.data, 1.)

    def forward(self, inp, mean=None, sigma=None):
        if mean is None:
            mean = 0.
        if sigma is None:
            sigma = 1.
        x = self.conv1(inp)
        x = self.bn(x)*sigma + mean
        x = self.relu(x)
        return x

class Conv3d(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3d, self).__init__()

        self.conv1 = nn.Conv3d(in_ch,
                               out_ch,
                               3, stride=stride,
                               padding=1)

        nn.init.xavier_uniform(self.conv1.weight.data, 1.)

    def forward(self, inp, mean=None, sigma=None):
        x = self.conv1(inp)
        return x


"""
  def forward(self, x, embedding=None):
    orig_x = x
    if self.with_coords:
      if self.is_3d:
        coords = self.coords.repeat(x.size(0), 1, 1, 1, 1)
      else:
        coords = self.coords.repeat(x.size(0), 1, 1, 1)
      x = torch.cat((x, coords), dim=1)
    if self.with_batchnorm:
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.bn2(self.conv2(out))
      if self.film is not None:
        out = self.film(out, embedding)
    else:
      out = self.conv2(F.relu(self.conv1(x)))
    res = orig_x if self.proj is None else self.proj(x)
    if self.with_residual:
      out = F.relu(res + out)
    else:
      out = F.relu(out)
    return out









"""


class ResBlockPost3d(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ResBlockPost3d, self).__init__()

        self.conv1 = nn.Conv3d(in_ch,
                               out_ch, 3, 1,
                               padding=1)
        self.bn = nn.InstanceNorm3d(out_ch, affine=True)
        self.relu = nn.ReLU(True)

        nn.init.xavier_uniform(self.conv1.weight.data, 1.)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn(x)
        x = self.relu(x)
        return x




class ResnetEncoderDecoder(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 input_size,
                 enc_dim,
                 ngf=64,
                 n_downsampling=None,
                 dense_code=False,
                 is_vae=True,
                 with_coords=True,
                 spatial_broadcast=False,
                 norm_layer=nn.BatchNorm2d):
        #assert(n_blocks >= 0)
        super(ResnetEncoderDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.is_vae = is_vae
        self.dense_code = dense_code
        self.with_coords = with_coords
        self.spatial_broadcast = spatial_broadcast
        self.input_size = input_size

        if self.spatial_broadcast and not self.dense_code:
            raise Exception("spatial_broadcast only compatible with dense codes")

        if is_vae:
            enc_dim = enc_dim*2

        n_extra = 0
        if self.with_coords:
            n_extra = 2

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        target_size = 2
        # How many downsamples do we need to do to go from
        # the input size to target size?
        if n_downsampling is None:
            n_downsampling = int(np.log(input_size / target_size) / np.log(2))-1

        # Need this for when dense_code=True
        self.enc_spatial_dim = input_size / (2**n_downsampling)

        self.preproc = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3,
                      bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        self.encoder = nn.ModuleList([])
        for i in range(n_downsampling):
            mult = 2**i
            n_in = ngf * mult
            if i == n_downsampling-1:
                if self.is_vae:
                    n_out = enc_dim*2
                else:
                    n_out = enc_dim
            else:
                n_out = ngf*mult*2
            self.encoder.append(
                nn.Sequential(nn.Conv2d(n_in+n_extra, n_out, kernel_size=3,
                                        stride=2, padding=1, bias=use_bias),
                              norm_layer(n_out),
                              nn.ReLU(True) if i != (n_downsampling-1) else nn.Identity())
            )

        mult = 2**n_downsampling

        if self.dense_code:
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.Identity()

        self.decoder = nn.ModuleList([])
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            if i == 0:
                n_ii = enc_dim
                n_jj = int(ngf*mult/2)
            else:
                n_ii = ngf*mult
                n_jj = int(ngf * mult / 2)

            if self.spatial_broadcast:
                # Spatial broadcast just does
                # non-strided convolutions
                self.decoder.append(nn.Sequential(
                    nn.Conv2d(n_ii+n_extra, n_jj,
                              kernel_size=3, stride=1,
                              padding=1,
                              bias=use_bias),
                    norm_layer(n_jj),
                    nn.ReLU(True)
                ))
            else:
                self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(n_ii+n_extra, n_jj,
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1,
                                       bias=use_bias),
                    norm_layer(n_jj),
                    nn.ReLU(True)
                ))

        self.post = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_jj, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, input):

        enc = self.encoder(input)
        dec = self.decoder(enc)
        return dec

    def coord_map(self, shape, start=-1, end=1):
        """
        Gives, a 2d shape tuple, returns two mxn coordinate maps,
        Ranging min-max in the x and y directions, respectively.
        """
        m, n = shape
        x_coord_row = torch.linspace(start, end, steps=n).type(torch.cuda.FloatTensor)
        y_coord_row = torch.linspace(start, end, steps=m).type(torch.cuda.FloatTensor)
        x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
        y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)
        return torch.cat([x_coords, y_coords], 0)

    def encode(self, input):

        x = self.preproc(input)
        for j in range(len(self.encoder)):
            cc = self.coord_map(x.shape[2:]).repeat(x.size(0), 1, 1, 1)
            x = self.encoder[j]( torch.cat((x,cc), dim=1) )

        if self.dense_code:
            enc = self.pool(x)
            enc = enc.view(-1, enc.size(1))
        else:
            enc = x

        if self.is_vae:
            enc_mu = enc[:, 0:(enc.size(1)//2)]
            enc_logvar = enc[:, (enc.size(1)//2):]
        else:
            enc_mu = enc
            # vae.py takes the exp of this, so
            # exp(-10) ~= 0.
            enc_logvar = torch.zeros_like(enc)-10.

        return enc_mu, enc_logvar

    def decode(self, input):

        if self.dense_code:
            input = input.view(-1, input.size(1), 1, 1)
            if self.spatial_broadcast:
                x = input.view(input.size(0), -1, 1, 1).\
                    repeat(1, 1, self.input_size, self.input_size)
            else:
                x = F.upsample(input, int(self.enc_spatial_dim))
        else:
            x = input

        for j in range(len(self.decoder)):
            cc = self.coord_map(x.shape[2:]).repeat(x.size(0), 1, 1, 1)
            x = self.decoder[j]( torch.cat((x,cc), dim=1) )

        x = self.post(x)

        return x



class ResnetEncoder(nn.Module):
    def __init__(self,
                 input_nc,
                 input_size,
                 enc_dim,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d):
        #assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf

        target_size = 2
        # How many downsamples do we need to do to go from
        # the input size to target size?
        n_downsampling = int(np.log(input_size / target_size) / np.log(2))

        self.enc_spatial_dim = input_size / (2**n_downsampling)

        encoder = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3,
                             ),
                   norm_layer(ngf),
                   nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2**i
            if i == n_downsampling-1:
                n_out = enc_dim*2
            else:
                n_out = ngf*mult*2
            encoder += [nn.Conv2d(ngf * mult, n_out, kernel_size=3,
                                  stride=2, padding=1)]
            if i != (n_downsampling-1):
                encoder += [
                    norm_layer(n_out),
                    nn.ReLU(True)
                ]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        enc = self.pool(self.encoder(input))
        return enc.view(-1, enc.size(1))




#####################################################################


###########

class HoloEncoderTest(nn.Module):
    def __init__(self,
                 input_nc,
                 ngf=64,
                 enc_dim=None,
                 norm_layer=nn.InstanceNorm3d,
                 norm_layer_2d=nn.InstanceNorm2d,
                 im_size=32):
        """
        """
        super(HoloEncoderTest, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf

        n_downsampling = 3

        encoder = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3,
                             bias=True),
                   norm_layer_2d(ngf),
                   nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2**i
            in_nf = ngf * mult
            if i==(n_downsampling-1) and enc_dim is not None:
                # `enc_dim` allows us to override the final
                # number of feature maps, otherwise it will
                # simply be ngf*(2**downsampling)`
                out_nf = enc_dim
            else:
                out_nf = ngf * mult * 2

            encoder += [nn.Conv2d(in_nf,
                                  (out_nf),
                                  kernel_size=3,
                                  stride=2, padding=1, bias=True)]
                        #norm_layer_2d((ngf * mult * 2) + theta_dim)]
            if i != n_downsampling-1:
                encoder += [norm_layer_2d((ngf * mult * 2)),
                            nn.ReLU(True)]
        self.encoder = nn.Sequential(*encoder)

    def encode(self, input):
        enc_ = self.encoder(input)
        return enc_


if __name__ == '__main__':
    net = ResnetEncoderDecoder(
        input_nc=3,
        output_nc=3,
        input_size=64,
        ngf=64,
        enc_dim=128,
        norm_layer=nn.InstanceNorm2d,
        resblock_after_stride=False
    )


    print(net)

    xfake = torch.randn((4,3,64,64))
    mu,sd = net.encode(xfake)
    dec = net.decode(mu)

    import pdb
    pdb.set_trace()
