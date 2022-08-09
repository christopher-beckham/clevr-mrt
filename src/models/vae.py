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

class VAE(Base):

    def __init__(self,
                 generator,
                 probe=None,
                 lamb=1.0,
                 beta=1.0,
                 prior_std=1.0,
                 cls_loss='cce',
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 update_g_every=1,
                 use_fp16=False,
                 handlers=[]):
        super(VAE, self).__init__()

        use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.generator = generator
        self.probe = probe

        self.lamb = lamb
        self.beta = beta

        self.prior_std = prior_std

        self.use_cuda = use_cuda

        if self.use_cuda:
            self.generator.to(self.device)
            if self.probe is not None:
                self.probe.to(self.device)

        optim_g = opt(filter(lambda p: p.requires_grad,
                             self.generator.parameters()), **opt_args)
        self.optim = {'g': optim_g}

        # Only add a probe optimiser IF it exists AND we don't backprop through it
        # with the autoencoder.
        if probe is not None:
            optim_probe = opt(filter(lambda p: p.requires_grad, probe.parameters()), **opt_args)
            self.optim['probe'] = optim_probe

        self.update_g_every = update_g_every

        self.schedulers = []
        #if scheduler_fn is not None:
        #    for key in self.optim:
        #        self.scheduler[key] = scheduler_fn(
        #            self.optim[key], **scheduler_args)
        self.handlers = handlers

        if cls_loss == 'cce':
            self.cls_loss_fn = nn.CrossEntropyLoss()
        elif cls_loss == 'mse':
            self.cls_loss_fn = self.mse
        else:
            raise Exception("Only cce or mse is currently supported for cls_loss")

        self.last_epoch = 0
        self.load_strict = True

        #torch.autograd.set_detect_anomaly(True)

    def _train(self):
        self.generator.train()
        if self.probe is not None:
            self.probe.train()

    def _eval(self):
        self.generator.eval()
        if self.probe is not None:
            self.probe.eval()

    def prepare_batch(self, batch):
        if len(batch) != 3:
            raise Exception("Expected batch to only contain two elements: " +
                            "X_batch, q_batch, and y_batch")
        X_batch = batch[0].float()
        q_batch = batch[1]
        y_batch = batch[2].flatten()
        if self.use_cuda:
            X_batch = X_batch.to(self.device)
            q_batch = q_batch.to(self.device)
            y_batch = y_batch.to(self.device)
        return [X_batch, q_batch, y_batch]

    def bce(self, prediction, target):
        if not hasattr(target, '__len__'):
            target = torch.ones_like(prediction)*target
            if prediction.is_cuda:
                target = target.to(self.device)
        loss = torch.nn.BCELoss()
        if prediction.is_cuda:
            loss = loss.to(self.device)
        target = target.view(-1, 1)
        return loss(prediction, target)

    def mse(self, prediction, target):
        return torch.mean((prediction-target)**2)

    def permute_dims(self, z):
        assert z.dim() == 2

        B, _ = z.size()
        perm_z = []
        for z_j in z.split(1, 1):
            perm = torch.randperm(B).to(z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)

        return torch.cat(perm_z, 1)

    def normal(self, mu, std):
        distn = Normal(mu, std)
        return distn, distn.rsample()

    def _normal_like(self, enc):
        return torch.zeros_like(enc).normal_(0., 1.)

    def train_on_instance(self,
                          x_batch,
                          q_batch,
                          y_batch,
                          **kwargs):
        self._train()
        for key in self.optim:
            self.optim[key].zero_grad()

        # ---------------
        # Reconstruction.
        # ---------------
        enc_mu, enc_logvar = self.generator.encode(x_batch)

        enc_dist, enc = self.normal(enc_mu, torch.exp(enc_logvar))
        dec_enc = self.generator.decode(enc)
        recon_loss = torch.mean(torch.abs(dec_enc-x_batch))

        with torch.no_grad():
            enc_logvar_mean = torch.mean(enc_logvar)
            enc_logvar_std = torch.std(enc_logvar)

        if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
            print("x shape:", x_batch.shape)
            print("enc shape:", enc.shape)
            print("recon shape:", dec_enc.shape)

        # ------------------------
        # KL divergence on z code.
        # ------------------------
        zeros = torch.zeros_like(enc_mu)
        ones = torch.zeros_like(enc_mu)+self.prior_std
        prior = Normal(zeros, ones)
        kl_z_loss = kl_divergence(enc_dist, prior).mean()

        gen_loss = self.lamb*recon_loss + \
            self.beta*kl_z_loss

        # -------------------------
        # FactorVAE generator loss.
        # -------------------------
        #if not self.disable_dh:
        #    fvae_g_loss = self.gen_loss_enc(enc_sampled)
        #    gen_loss = gen_loss + self.sigma*fvae_g_loss

        gen_loss.backward()

        self.optim['g'].step()

        ## -------------------
        ## Probe, if it exists
        ## -------------------

        if self.probe is not None:
            self.optim['probe'].zero_grad()

            probe_out = self.probe(enc.detach(), q_batch)
            probe_loss = self.cls_loss_fn(probe_out, y_batch)
            probe_loss.backward()
            with torch.no_grad():
                if self.cls_loss_fn != self.mse:
                    probe_acc = (probe_out.argmax(dim=1).long() == y_batch).float().mean()
                else:
                    probe_acc = probe_loss
            self.optim['probe'].step()

        losses = {
            'gen_loss': gen_loss.item(),
            'recon': recon_loss.item(),
            'kl_z': kl_z_loss.item(),
            'enc_logvar_mean': enc_logvar_mean.item(),
            'enc_logvar_std': enc_logvar_std.item()
        }
        #if not self.disable_dh:
        #    losses['fvae'] = fvae_loss.item()
        #    losses['fvae_g'] = fvae_g_loss.item()

        if self.probe is not None:
            losses['probe_loss'] = probe_loss.item()
            losses['probe_acc'] = probe_acc.item()

        outputs = {
            'recon': dec_enc,
            'input': x_batch,
        }
        return losses, outputs

    def eval_on_instance(self,
                         x_batch,
                         q_batch,
                         y_batch,
                         **kwargs):
        self._eval()
        losses = {}
        with torch.no_grad():

            enc_mu, enc_logvar = self.generator.encode(x_batch)
            enc = self.normal(enc_mu, torch.exp(enc_logvar))[1]
            dec_enc = self.generator.decode(enc)
            recon_loss = torch.mean(torch.abs(dec_enc-x_batch))

            losses['recon_loss'] = recon_loss.item()

            if self.probe is not None:
                # y_batch contains another camera view of
                # x_batch.

                probe_out = self.probe(enc.detach(), q_batch)
                probe_loss = self.cls_loss_fn(probe_out, y_batch)
                if self.cls_loss_fn != self.mse:
                    probe_acc = (probe_out.argmax(dim=1).long() == y_batch).float().mean()
                else:
                    probe_acc = probe_loss

                losses['probe_loss'] = probe_loss.item()
                losses['probe_acc'] = probe_acc.item()

        outputs = {
            'recon': dec_enc,
            'input': x_batch,
        }
        return losses, outputs

    def save(self, filename, epoch):
        dd = {}
        # Save the models.
        dd['g'] = self.generator.state_dict()
        # Save probe, if it exists
        if self.probe is not None:
            dd['probe'] = self.probe.state_dict()
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
        self.generator.load_state_dict(dd['g'], strict=self.load_strict)
        if self.probe is not None:
            self.probe.load_state_dict(dd['probe'], strict=self.load_strict)
        # Load the models' optim state.
        for key in self.optim:
            if ('optim_%s' % key) in dd:
                self.optim[key].load_state_dict(dd['optim_%s' % key])
        self.last_epoch = dd['epoch']
