import torch
from torch import nn

from .networks import (ResBlock3d,
                       ConvBlock2d,
                       ConvBlockUpsample2d,
                       ConvBlock3d)



class VolumeNetSpatialBroadcast(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 nvf=None,
                 vs=4,
                 n_ds=5,
                 v_ds=[4, 8],
                 enc_dim=None,
                 norm_layer=nn.InstanceNorm3d,
                 norm_layer_2d=nn.InstanceNorm2d,
                 im_size=32):
        """
        Using spatial broadcast style:
        """

        super(VolumeNetSpatialBroadcast, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        if nvf is not None:
            raise Exception("`nvf` has been replaced with enc_dim in this class")

        # Previously, n_ds=5 for 64px images
        # and n_ds=3 for 32px images.
        n_downsampling = n_ds

        #################
        # Angle encoder #
        #################
        theta_dim = 12
        div_factor = 2
        theta = [nn.Conv2d(input_nc, ngf // div_factor, kernel_size=7, padding=3,
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

        ############################################
        # Encode the image into a tight bottleneck #
        ############################################

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
                out_nf = enc_dim*2
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
        encoder += [nn.AdaptiveAvgPool2d(1)]



        ###################################
        # Define the volume to be learned #
        ###################################

        #vol = torch.ones((nvf, vs, vs, vs)) # 4x4x4

        # 8x8x8, 16x16x16

        #torch.nn.init.xavier_normal_(vol)
        #self.vol = nn.Parameter(vol, requires_grad=True)

        ###################################################
        # Define the decoder which modifies vol via ADAIN #
        ###################################################

        self.relu = nn.ReLU()
        self.ups_3d = nn.Upsample(scale_factor=2, mode='nearest')

        # ups here

        # e.g. 1024 -> (nvf*2)*2 for mean+var
        self.conv3d_1 = ResBlock3d(nvf, nvf//v_ds[0])
        self.conv3d_1b = ConvBlock3d(nvf//v_ds[0], nvf//v_ds[0])

        self.conv3d_2 = ResBlock3d(nvf//v_ds[0], nvf//v_ds[1])
        self.conv3d_2b = ConvBlock3d(nvf//v_ds[1], nvf//v_ds[1])

        # e.g. 256*2 * (4*2) = 512*8 = 4096
        ngf_new = (nvf//v_ds[-1]) * (vs*(2**2))

        #self.conv3d_post = ResBlockPost3d(nvf//v_ds[1], nvf//v_ds[1])

        #self.proj = nn.Conv2d(ngf_new, ngf_new, kernel_size=1)

        #print(ngf_new)

        # projection unit
        # this part takes up a lot of model capacity, so we could
        # do a 1x1 projection here to save on space.

        decoder = []

        # If we start off with a template volume of `vs`
        # and a desired img size of `im_size`, how many
        # decoder blocks do we need to use?
        n_decodes = int(np.log(im_size/ (vs*4)) / np.log(2))


        for i in range(n_decodes):
            j = 2**i
            jj = 2**(i+1)
            # Stride conv then proc conv
            decoder.append(ConvBlockUpsample2d(ngf_new // j, ngf_new // jj))
            decoder.append(ConvBlock2d(ngf_new // jj, ngf_new // jj))

        post = []
        post += [nn.ReflectionPad2d(3)]
        post += [nn.Conv2d(ngf_new // jj,
                           output_nc,
                           kernel_size=7,
                           padding=0)]
        post += [nn.Tanh()]

        self.encoder = nn.Sequential(*encoder)
        self.theta = nn.Sequential(*theta)
        #self.decoder = nn.Sequential(*decoder)
        self.post = nn.Sequential(*post)

        self.dec = nn.Sequential(*decoder)

    def forward(self, input):
        enc, h = self.encoder(input)
        dec = self.decoder(enc, h)
        return dec

    def enc2vol(self, enc):

        # (bs, enc_dim, vs, vs, vs)
        h = enc.view(enc.size(0), -1, 1, 1, 1).\
            repeat(1, 1, self.vs, self.vs, self.vs)
        #h = self.vol.repeat((enc.size(0), 1, 1, 1, 1))
        # resblock1
        h = self.conv3d_1(h, None, None)
        h = self.conv3d_1b(h)
        # resblock2
        h = self.conv3d_2(h, None, None)
        h = self.conv3d_2b(h)

        return h

    def encode(self, input):
        # Encode the image into two halves:
        # the deterministic component, and
        # the stochastic one.
        enc_ = self.encoder(input)
        enc_ = enc_.view(-1, enc_.size(1))

        enc_mu = enc_[:, 0:(enc_.size(1)//2)]
        enc_std = enc_[:, (enc_.size(1)//2):]

        # Also extract the predicted angle.
        theta = self.theta(input)
        theta = theta.view(-1, theta.size(1))

        return enc_mu, enc_std, theta

    def decode(self, h):

        # Post-processing block here,
        # otherwise we see the zero padding
        # when we decode.
        #h = self.conv3d_post(h)

        h = h.view(-1, h.size(1)*h.size(2), h.size(3), h.size(4))
        # Ok, now decode in 2d but keep augmenting it with
        # the encoding.
        h = self.dec(h)
        h = self.post(h)
        return h
