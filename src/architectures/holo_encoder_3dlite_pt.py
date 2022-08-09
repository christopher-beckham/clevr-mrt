import torch
from torch import nn
import torch.nn.functional as F
from .shared import networks
from functools import partial
from itertools import chain

from .shared.networks import ConvBlock3d
from .holo_encoder_base import HoloEncoderBase

def get_resnet():
    from torchvision.models import resnet101
    enc = resnet101(pretrained=True)
    layers = [
        enc.conv1,
        enc.bn1,
        enc.relu,
        enc.maxpool,
    ]
    for i in range(3):
        layer_name = 'layer%d' % (i + 1)
        layers.append(getattr(enc, layer_name))
    enc = torch.nn.Sequential(*layers)
    enc.cuda()
    enc.eval()
    for param in enc.parameters():
        param.requires_grad = False
    return enc

class HoloEncoder3dLitePretrained(HoloEncoderBase):
    """

    """

    def train(self):
        # We override it here because the
        # pre-trained resnet must always
        # be in eval, and everything else
        # must conform to train/eval.
        self.imagenet.eval()
        self.cam_encode.train()
        self.post_encode.train()
        #print("imagenet: eval, rest: train")

    def eval(self):
        # We override it here because the
        # pre-trained resnet must always
        # be in eval, and everything else
        # must conform to train/eval.
        self.imagenet.eval()
        self.cam_encode.eval()
        self.post_encode.eval()
        #print("imagenet: eval, rest: eval")

    def __init__(self,
                 input_nc,
                 output_nc,
                 target_nf=2048,
                 theta_dim=32,
                 freeze_postproc=False,
                 n_postproc=0,
                 norm_layer=nn.InstanceNorm3d,
                 norm_layer_2d=nn.InstanceNorm2d,
                 im_size=32):
        """
        """
        super(HoloEncoder3dLitePretrained, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.imagenet = get_resnet()

        # Take the camera coordinates and
        # map them to a rotation matrix.
        self.cam_encode = nn.Sequential(
            nn.Linear(6, theta_dim),
            nn.BatchNorm1d(theta_dim),
            nn.ReLU(),
            nn.Linear(theta_dim, 6)
        )

        out_nf = 1024
        self.post_encode = nn.Sequential(
            nn.Conv2d(out_nf, target_nf, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(target_nf),
            nn.ReLU(True),
            nn.Conv2d(target_nf, target_nf, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(target_nf),
            nn.ReLU(True)
        )
        if n_postproc > 0:
            postproc = []
            for k in range(n_postproc):
                postproc.append(
                    ConvBlock3d(
                        target_nf // 16,
                        target_nf // 16,
                        norm_layer=norm_layer
                    )
                )
            self.postproc = nn.ModuleList(postproc)
        else:
            self.postproc = nn.ModuleList(
                [nn.Identity()]*n_postproc
            )
        if freeze_postproc:
            print("freeze_postproc=True, so freezing postprocessor params...")
            for p in chain(self.postproc.parameters(),
                           self.post_encode.parameters()):
                p.requires_grad = False

    def forward(self, input):
        #enc, h = self.encoder(input)
        #dec = self.decoder(enc, h)
        #return dec
        return None

    def enc2vol(self, x):
        # e.g. (1024,16,16)
        # goes to (64,16,16,16)
        h = x.view(-1,
                   x.size(1)//16,
                   16,
                   x.size(2),
                   x.size(3))
        for pp in self.postproc:
            h = pp(h, None, None)
        return h

    def encode(self, input):

        return self.post_encode(
            self.imagenet(input).detach()
        )

def get_network(n_channels,
                theta_dim=32,
                target_nf=2048,
                freeze_postproc=False,
                n_postproc=0,
                im_size=224,
                use_bn=False):
    if use_bn:
        norm_layer = partial(nn.BatchNorm3d, affine=True)
        norm_layer_2d = partial(nn.BatchNorm2d, affine=True)
    else:
        norm_layer = partial(nn.InstanceNorm3d, affine=True)
        norm_layer_2d = partial(nn.InstanceNorm2d, affine=True)
    gen = HoloEncoder3dLitePretrained(input_nc=n_channels,
                                      output_nc=n_channels,
                                      theta_dim=theta_dim,
                                      target_nf=target_nf,
                                      freeze_postproc=freeze_postproc,
                                      n_postproc=n_postproc,
                                      norm_layer=norm_layer,
                                      norm_layer_2d=norm_layer_2d,
                                      im_size=im_size)
    return {
        'gen': gen
    }
