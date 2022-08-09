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

class DeepInfoMaxLoss(nn.Module):
    def __init__(self,
                 global_d, local_d, prior_d,
                 alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.global_d = global_d
        self.local_d = local_d
        self.prior_d = prior_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, M.size(2), M.size(3))

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR

class DIM(Base):

    def __init__(self,
                 enc,
                 dg,
                 dl,
                 dp,
                 probe=None,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 prior_std=1.0,
                 sup=False,
                 cls_loss='cce',
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 handlers=[]):
        super(DIM, self).__init__()

        use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.enc = enc
        self.dg = dg
        self.dl = dl
        self.dp = dp
        self.probe = probe

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Do we backprop through the probe?
        self.sup = sup
        self.prior_std = prior_std

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.enc.to(self.device)
            self.dg.to(self.device)
            self.dl.to(self.device)
            self.dp.to(self.device)
            if self.probe is not None:
                self.probe.to(self.device)

        optim_enc = opt(filter(lambda p: p.requires_grad,
                               self.enc.parameters()), **opt_args)
        optim_dg = opt(filter(lambda p: p.requires_grad,
                              self.dg.parameters()), **opt_args)
        optim_dl = opt(filter(lambda p: p.requires_grad,
                              self.dl.parameters()), **opt_args)
        optim_dp = opt(filter(lambda p: p.requires_grad,
                              self.dp.parameters()), **opt_args)

        self.loss = DeepInfoMaxLoss(global_d=dg,
                                    local_d=dl,
                                    prior_d=dp,
                                    alpha=alpha,
                                    beta=beta,
                                    gamma=gamma)

        self.optim = {
            'enc': optim_enc,
            'dg': optim_dg,
            'dl': optim_dl,
            'dp': optim_dp
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
        self.dg.train()
        self.dl.train()
        self.dp.train()

    def _eval(self):
        self.enc.eval()
        self.dg.eval()
        self.dl.eval()
        self.dp.eval()

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

        y, M = self.enc(x_batch)

        if kwargs['iter']==1 and kwargs['epoch']==1:
            print("y shape:", y.shape)
            print("M shape:", M.shape)

        M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)

        loss = self.loss(y, M, M_prime)

        loss.backward()
        for key in self.optim:
            if key == 'probe':
                continue
            self.optim[key].step()

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

            y, M = self.enc(x_batch)

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
        dd['dg'] = self.dg.state_dict()
        dd['dl'] = self.dl.state_dict()
        dd['dp'] = self.dp.state_dict()
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
        self.dg.load_state_dict(dd['dg'], strict=self.load_strict)
        self.dl.load_state_dict(dd['dl'], strict=self.load_strict)
        self.dp.load_state_dict(dd['dp'], strict=self.load_strict)
        if self.probe is not None:
            self.probe.load_state_dict(dd['probe'], strict=self.load_strict)
        # Load the models' optim state.
        for key in self.optim:
            if ('optim_%s' % key) in dd:
                self.optim[key].load_state_dict(dd['optim_%s' % key])
        self.last_epoch = dd['epoch']
