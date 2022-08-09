import torch
from torch import nn
import torch.nn.functional as F
from .util import Flatten

class Probe(nn.Module):
    def __init__(self, n_in, nf, n_layers, n_classes, mode):
        super(Probe, self).__init__()
        layers = []
        for j in range(n_layers):
            if j == 0:
                n_in_ = n_in
            else:
                n_in_ = nf

            if j != (n_layers-1):
                n_out = nf
            else:
                n_out = n_classes
            layers += [
                nn.Linear(n_in_, n_out),
            ]
            if j != (n_layers-1):
                layers += [nn.BatchNorm1d(n_out), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        if mode not in ['all', 'angles', 'trans']:
            raise Exception("Mask mode must be either `all`, `angles`, or `trans`")
        self.mode = mode

    def forward(self, h, q, enc):
        mask = torch.zeros_like(q)
        if self.mode == 'all':
            # Use the entire q code
            mask += 1.
        elif self.mode == 'angles':
            # Only use the angles (mask out the)
            # translation.
            mask[:, 0:6] += 1.
        elif self.mode == 'trans':
            # Only use the translation (mask out the angles)
            mask[:, 6:12] += 1.

        return self.mlp(q*mask)

def get_network(n_in=12, nf=128, n_layers=2, n_classes=136, mode='all'):
    return Probe(n_in=n_in,
                 nf=nf,
                 n_layers=n_layers,
                 n_classes=n_classes,
                 mode=mode)
