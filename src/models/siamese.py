"""
Some code taken from here: https://github.com/DuaneNielsen/DeepInfomaxPytorch
"""

import torch
import numpy as np
from collections import OrderedDict
from torch import optim
from torch.nn import functional as F
from itertools import chain
from .base import Base
from torch import nn
from torch.distributions import (Normal,
                                 kl_divergence)

class Siamese(Base):
    """
    Intended for contrastive self-supervised learning
    application.

    """

    def __init__(self,
                 enc,
                 probe=None,
                 prior_std=1.0,
                 sup=False,
                 cls_loss='cce',
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 handlers=[]):
        super(Siamese, self).__init__()

        use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.enc = enc
        self.probe = probe

        # Do we backprop through the probe?
        self.sup = sup
        self.prior_std = prior_std

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.enc.to(self.device)
            if self.probe is not None:
                self.probe.to(self.device)

        optim_enc = opt(filter(lambda p: p.requires_grad,
                               self.enc.parameters()), **opt_args)

        # <anchor, pos, negative>
        self.loss = nn.TripletMarginLoss()

        self.optim = {
            'enc': optim_enc,
        }

        if cls_loss == 'cce':
            self.cls_loss_fn = nn.CrossEntropyLoss()
        elif cls_loss == 'mse':
            self.cls_loss_fn = self.mse
        else:
            raise Exception("Only cce or mse is currently supported for cls_loss")

        # Only add a probe optimiser IF it exists AND we don't backprop through it
        # with the autoencoder.

        if probe is not None:
            optim_probe = opt(filter(lambda p: p.requires_grad, probe.parameters()), **opt_args)
            self.optim['probe'] = optim_probe

        self.schedulers = []
        self.handlers = handlers

        self.last_epoch = 0
        self.load_strict = True

    def _train(self):
        self.enc.train()
        if self.probe is not None:
            self.probe.train()

    def _eval(self):
        self.enc.eval()
        if self.probe is not None:
            self.probe.eval()

    def mse(self, prediction, target):
        return torch.mean((prediction-target)**2)

    def prepare_batch(self, batch):
        if len(batch) != 3:
            raise Exception("Expected batch to only contain two elements: " +
                            "X_batch, meta_batch, and y_batch")
        X_batch = batch[0].float()
        meta_batch = batch[1]
        y_batch = batch[2]
        # If meta batch is a list/tuple, just make a new
        # tuple putting them on the GPU. Otherwise, if it's
        # just a single tensor put it on the GPU.
        if type(meta_batch) in [tuple, list]:
            meta_batch = tuple([x.to(self.device) for x in meta_batch])
        else:
            meta_batch = (meta_batch.to(self.device),)
        # Label
        #y_batch = y_batch.long()
        # Cuda
        if self.use_cuda:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
        return [X_batch, meta_batch, y_batch]

    def train_on_instance(self,
                          x_batch,
                          meta_batch,
                          y_batch,
                          **kwargs):
        self._train()
        for key in self.optim:
            self.optim[key].zero_grad()

        enc_anchor = self.enc(x_batch)
        enc_pos = self.enc(y_batch)
        perm = torch.randperm(y_batch.size(0))
        enc_neg = self.enc(y_batch[perm])

        # (enc_x, enc_y) is a +ve pair
        # (enc_x, shuffle(enc_x)) is a -ve pair

        if kwargs['iter']==1 and kwargs['epoch']==1:
            print("enc_anchor shape:", enc_anchor.shape)

        loss = self.loss(enc_anchor,
                         enc_pos,
                         enc_neg)
        loss.backward()

        self.optim['enc'].step()

        if self.probe is not None:
            # Only train the probe after the specified
            # amount of epochs.
            self.optim['probe'].zero_grad()
            # The classifier takes the latent volume h
            # and encoding enc, along with any elements
            # in `meta_batch`.
            probe_out = self.probe(y.detach())
            probe_loss = self.cls_loss_fn(probe_out, y_batch)
            probe_loss.backward()
            with torch.no_grad():
                if self.cls_loss_fn != self.mse:
                    probe_acc = (probe_out.argmax(dim=1).long() == y_batch).float().mean()
                else:
                    probe_acc = probe_loss
            self.optim['probe'].step()

        losses = {
            'loss': loss.item()
        }

        if self.probe is not None:
            losses['probe_loss'] = probe_loss.item()
            losses['probe_acc'] = probe_acc.item()

        outputs = {
        }
        return losses, outputs

    def eval_on_instance(self,
                         x_batch,
                         meta_batch,
                         y_batch,
                         **kwargs):
        self._eval()
        losses = {}
        with torch.no_grad():

            enc_anchor = self.enc(x_batch)
            enc_pos = self.enc(y_batch)
            perm = torch.randperm(y_batch.size(0))
            enc_neg = self.enc(y_batch[perm])

            loss = self.loss(enc_anchor,
                             enc_pos,
                             enc_neg)

            losses['loss'] = loss.item()

            if self.probe is not None:
                probe_out = self.probe(y)
                probe_loss = self.cls_loss_fn(probe_out, y_batch)
                if self.cls_loss_fn != self.mse:
                    probe_acc = (probe_out.argmax(dim=1).long() == y_batch).float().mean()
                else:
                    probe_acc = probe_loss
                losses['probe_loss'] = probe_loss.item()
                losses['probe_acc'] = probe_acc.item()

        return losses, {}

    def save(self, filename, epoch):
        dd = {}
        # Save the models.
        dd['enc'] = self.enc.state_dict()
        # Save the models' optim state.
        for key in self.optim:
            dd['optim_%s' % key] = self.optim[key].state_dict()
        dd['epoch'] = epoch
        torch.save(dd, filename)

    def load(self, filename):
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        dd = torch.load(filename,
                        map_location=map_location)
        # Load the models.
        self.enc.load_state_dict(dd['enc'], strict=self.load_strict)
        if self.probe is not None:
            self.probe.load_state_dict(dd['probe'], strict=self.load_strict)
        # Load the models' optim state.
        for key in self.optim:
            if ('optim_%s' % key) in dd:
                self.optim[key].load_state_dict(dd['optim_%s' % key])
        self.last_epoch = dd['epoch']

########################################################
########################################################
########################################################

class SiameseIQTT(Siamese):
    """
    Specifically for the IQTT dataset, since it's a bit
    esoteric.
    """

    def __init__(self, *args, **kwargs):
        #self.sigma = kwargs.pop('sigma')
        super(SiameseIQTT, self).__init__(*args, **kwargs)
        #if not hasattr(self.cls, 'embed') and \
        #   not hasattr(self.cls, 'out'):
        #    raise Exception("The given network must have the two methods: " +
        #                    "`embed` and `out`.")
        self.loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def prepare_batch(self, batch):
        if len(batch) != 2:
            raise Exception("Expected batch to only contain X_pos, X_neg")
        Xp_batch = batch[0].float()
        Xn_batch = batch[1].float()
        if self.use_cuda:
            Xp_batch = Xp_batch.cuda()
            Xn_batch = Xn_batch.cuda()
        return [Xp_batch, Xn_batch]

    def _noise(self, x):
        noise = torch.zeros_like(x).normal_(0, self.sigma)
        if x.is_cuda:
            noise = noise.cuda()
        return x+noise

    def train_on_instance(self,
                          xp_batch,
                          xn_batch,
                          **kwargs):
        self._train()
        self.optim['cls'].zero_grad()

        xp1, xp2 = self._noise(xp_batch[:, 0:3]), self._noise(xp_batch[:, 3:6])
        xn = self._noise(xn_batch[:, 3:6])

        embed_xp1 = self.enc(xp1)
        embed_xp2 = self.enc(xp2)
        embed_xn = self.enc(xn)

        # loss(anchor, pos, neg)
        loss = self.loss(embed_xp1, embed_xp2, embed_xn)

        with torch.no_grad():
            dist_pos = torch.mean((embed_xp1-embed_xp2)**2)
            dist_neg = torch.mean((embed_xp1-embed_xn)**2)

        loss.backward()
        self.optim['cls'].step()

        losses = {}
        losses['loss'] = loss.item()
        losses['dist_pos'] = dist_pos.item()
        losses['dist_neg'] = dist_neg.item()

        outputs = {
        }

        return losses, outputs

    def eval_on_instance(self,
                         xp_batch,
                         xn_batch,
                         **kwargs):
        self._eval()

        with torch.no_grad():

            xp1, xp2 = xp_batch[:, 0:3], xp_batch[:, 3:6]
            xn = xn_batch[:, 3:6]

            embed_xp1 = self.enc(xp1)
            embed_xp2 = self.enc(xp2)
            embed_xn = self.enc(xn)

            # loss(anchor, pos, neg)
            loss = self.loss(embed_xp1, embed_xp2, embed_xn)

            with torch.no_grad():
                dist_pos = torch.mean((embed_xp1-embed_xp2)**2)
                dist_neg = torch.mean((embed_xp1-embed_xn)**2)

            return {
                'loss': loss.item(),
                'dist_pos': dist_pos.item(),
                'dist_neg': dist_neg.item()
            }, {}
 
