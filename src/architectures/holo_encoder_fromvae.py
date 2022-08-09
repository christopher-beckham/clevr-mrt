import torch
from torch import nn
import torch.nn.functional as F
from .shared import networks
from . import discriminators
from functools import partial

class HoloEncoderFromVae(nn.Module):
    """
    To be used with models/holo_encoder.py and
    with a pre-trained HoloVAE model.
    """
    def __init__(self,
                 encoder,
                 theta_dim=32):
        """
        """
        super(HoloEncoderFromVae, self).__init__()

        self.encoder = encoder
        # Just to be sure, zip through params and make
        # requires_grad=False for all params, and put
        # thing in eval mode.
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.cuda()

        # Take the camera coordinates and
        # map them to a rotation matrix.
        self.cam_encode = nn.Sequential(
            nn.Linear(6, theta_dim),
            nn.BatchNorm1d(theta_dim),
            nn.ReLU(),
            nn.Linear(theta_dim, 6)
        )
        self.cam_encode.cuda()

    def encode(self, input):
        z, _ = self.encoder.encode(input)
        return z.detach()

    def enc2vol(self, z):
        return self.encoder.enc2vol(z).detach()

    def decode(self, h):
        return self.encoder.decode(h)
