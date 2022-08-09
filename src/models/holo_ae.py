import torch
import numpy as np
from collections import OrderedDict
from torch import optim
from torch.nn import functional as F
from itertools import chain
from .base import Base
from torch import nn
from torch.distributions import (Normal,
                                 Uniform,
                                 kl_divergence)
from .util.ssim import SSIM

class HoloAE(Base):

    def _to_radians(self, deg):
        return deg * (np.pi / 180)

    def __init__(self,
                 generator,
                 disc_x,
                 disc_enc,
                 probe=None,
                 lamb=1.0,
                 beta=1.0,
                 gamma=1.0,
                 sigma=1.0,
                 eps=1.0,
                 disable_gan=False,
                 disable_dh=False,
                 disable_multiview=False,
                 blanking_loss=False,
                 pad_rotate=False,
                 interp_mode='trilinear',
                 projection='perspective',
                 supervised=False,
                 fix_t=None,
                 fix_n=None,
                 gan_loss='bce',
                 cls_loss='cce',
                 prior_std=1.0,
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 update_g_every=1,
                 use_fp16=False,
                 handlers=[]):
        super(HoloAE, self).__init__()

        use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if projection not in ['perspective',
                              'perspective2',
                              'perspective3',
                              'orthogonal']:
            raise Exception("`projection` must be either: " + \
                            "'perspective', 'perspective2', or 'orthogonal'")

        self.generator = generator
        self.disc_x = disc_x
        self.disc_enc = disc_enc
        self.probe = probe

        self.disable_multiview = disable_multiview

        self.projection = projection
        self.interp_mode = interp_mode

        self.lamb = lamb
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.sigma = sigma

        self.prior_std = prior_std

        self.disable_gan = disable_gan
        self.disable_dh = disable_dh

        self.supervised = supervised

        self.fix_n = fix_n
        self.fix_t = fix_t


        #self.min_angle = -2.
        #self.max_angle = 2.
        self.min_angle = -prior_std * 2.
        self.max_angle = prior_std * 2.

        self.use_fp16 = use_fp16

        self.rot2idx = {
            'x': 0,
            'z': 1,
            'y': 2
        }

        self.use_cuda = use_cuda

        if self.use_cuda:
            self.generator.to(self.device)
            if self.disc_x is not None:
                self.disc_x.to(self.device)
            if self.disc_enc is not None:
                self.disc_enc.to(self.device)
            if self.probe is not None:
                self.probe.to(self.device)

        optim_g = opt(filter(lambda p: p.requires_grad,
                             self.generator.parameters()), **opt_args)

        # Apex library options.
        """
        if use_fp16:
            if self.disable_gan is False:
                raise Exception("fp16 does not support GAN mode")
            self.opt_level = 'O1'
        else:
            self.opt_level = 'O0'
        self.generator, optim_g = amp.initialize(self.generator,
                                                 optim_g,
                                                 opt_level=self.opt_level,
                                                 enabled=use_fp16)
        """
        self.optim = {'g': optim_g}

        if self.disc_x is not None:
            optim_disc_x = opt(filter(lambda p: p.requires_grad,
                                      self.disc_x.parameters()), **opt_args)
            self.optim['disc_x'] = optim_disc_x

        if self.disc_enc is not None:
            optim_disc_enc = opt(filter(lambda p: p.requires_grad,
                                        self.disc_enc.parameters()), **opt_args)
            self.optim['disc_enc'] = optim_disc_enc

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

        ##################
        # Loss functions #
        ##################
        if gan_loss == 'bce':
            self.gan_loss_fn = self.bce
        elif gan_loss == 'mse':
            self.gan_loss_fn = self.mse
        else:
            raise Exception("Only bce or mse is currently supported for gan_loss")

        if cls_loss == 'cce':
            self.cls_loss_fn = nn.CrossEntropyLoss()
        elif cls_loss == 'mse':
            self.cls_loss_fn = self.mse
        else:
            raise Exception("Only cce or mse is currently supported for cls_loss")


        self.last_epoch = 0
        self.load_strict = True

        #torch.autograd.set_detect_anomaly(True)


    def dis_loss(self, out_real, out_fake):
        #r_preds, r_mus, r_sigmas = self.dis(real_samps)
        #f_preds, f_mus, f_sigmas = self.dis(fake_samps)
        r_preds = self.disc_x(out_real)[0]
        f_preds = self.disc_x(out_fake)[0]

        loss = (torch.mean(torch.nn.ReLU()(1 - r_preds)) +
                torch.mean(torch.nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, out_fake):
        return -torch.mean(self.disc_x(out_fake)[0])

    def dis_loss_enc(self, out_real, out_fake):
        r_preds = self.disc_enc(out_real)
        f_preds = self.disc_enc(out_fake)

        loss = (torch.mean(torch.nn.ReLU()(1 - r_preds)) +
                torch.mean(torch.nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss_enc(self, out_fake):
        return -torch.mean(self.disc_enc(out_fake))

    def train(self):
        self._train()

    def eval(self):
        self._eval()

    def _train(self):
        self.generator.train()
        if self.disc_x is not None:
            self.disc_x.train()
        if self.disc_enc is not None:
            self.disc_enc.train()
        if self.probe is not None:
            self.probe.train()

    def _eval(self):
        self.generator.eval()
        if self.disc_x is not None:
            self.disc_x.eval()
        if self.disc_enc is not None:
            self.disc_enc.eval()
        if self.probe is not None:
            self.probe.eval()

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

    def reconstruct(self, x_batch):
        """Get reconstruction.

        :param x_batch: 
        :returns: 
        :rtype: 

        """
        with torch.no_grad():
            enc, q, h = self.generator.encode(x_batch)
            rz, ry, rx, tx, ty, tz = self._extract_angles_and_offsets(q)
            h_rot = self.rotate(h,
                                rz[1], ry[1], rx[1],
                                tx[1], ty[1], tz[1])
            dec_enc = self.generator.decode(h_rot, enc)
            return dec_enc

    def t_eye(self, bs):
        """
        """
        mat = torch.zeros((bs, 3, 4)).float().cuda()
        mat[:, 0, 0] = 1.
        mat[:, 1, 1] = 1.
        mat[:, 2, 2] = 1.
        return mat

    def t_rot_matrix_x(self, theta):
        """
        theta: measured in radians
        """
        bs = theta.size(0)
        mat = torch.zeros((bs, 3, 3)).float()
        if theta.is_cuda:
            mat = mat.cuda()
        mat[:, 0, 0] = 1.
        mat[:, 1, 1] = torch.cos(theta).view(-1)
        mat[:, 1, 2] = -torch.sin(theta).view(-1)
        mat[:, 2, 1] = torch.sin(theta).view(-1)
        mat[:, 2, 2] = torch.cos(theta).view(-1)
        return mat

    def t_rot_matrix_y(self, theta):
        """
        theta: measured in radians
        """
        bs = theta.size(0)
        mat = torch.zeros((bs, 3, 3)).float()
        if theta.is_cuda:
            mat = mat.cuda()
        mat[:, 0, 0] = torch.cos(theta).view(-1)
        mat[:, 0, 2] = torch.sin(theta).view(-1)
        mat[:, 1, 1] = 1.
        mat[:, 2, 0] = -torch.sin(theta).view(-1)
        mat[:, 2, 2] = torch.cos(theta).view(-1)
        return mat

    def t_rot_matrix_z(self, theta):
        """
        theta: measured in radians
        """
        bs = theta.size(0)
        mat = torch.zeros((bs, 3, 3)).float()
        if theta.is_cuda:
            mat = mat.cuda()
        mat[:, 0, 0] = torch.cos(theta).view(-1)
        mat[:, 0, 1] = -torch.sin(theta).view(-1)
        mat[:, 1, 0] = torch.sin(theta).view(-1)
        mat[:, 1, 1] = torch.cos(theta).view(-1)
        mat[:, 2, 2] = 1.
        return mat

    def sample_angles(self,
                      bs,
                      min_angle,
                      max_angle):
        """Sample random yaw, pitch, and roll angles"""
        angles = []
        for i in range(bs):
            rnd_angles = [
                np.random.uniform(min_angle, max_angle),
            ]
            angles.append(rnd_angles)
        return np.asarray(angles)

    def sample(self, x_batch):
        """Output a random sample (a rotation)
        :param x_batch: the batch to randomly rotate.
        :returns:
        :rtype:
        """
        raise NotImplementedError()

    def permute_dims(self, z):
        assert z.dim() == 2

        B, _ = z.size()
        perm_z = []
        for z_j in z.split(1, 1):
            perm = torch.randperm(B).to(z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)

        return torch.cat(perm_z, 1)

    def interpolate_trilinear(self, img, x, y, z):

        # ...

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        z0 = torch.floor(z).long()
        z1 = z0 + 1

        ix = torch.zeros_like(x0)
        for j in range(x.size(0)):
            ix[j] += j

        x0 = torch.clamp(x0, min=0, max=img.shape[2] - 1)
        x1 = torch.clamp(x1, min=0, max=img.shape[2] - 1)
        y0 = torch.clamp(y0, min=0, max=img.shape[3] - 1)
        y1 = torch.clamp(y1, min=0, max=img.shape[3] - 1)
        z0 = torch.clamp(z0, min=0, max=img.shape[4] - 1)
        z1 = torch.clamp(z1, min=0, max=img.shape[4] - 1)

        x_ = x - x0.float()
        y_ = y - y0.float()
        z_ = z - z0.float()

        out = (img[:, ix, x0, y0, z0]*(1-x_)*(1-y_)*(1-z_) +
               img[:, ix, x1, y0, z0]*x_*(1-y_)*(1-z_) +
               img[:, ix, x0, y1, z0]*(1-x_)*y_*(1-z_) +
               img[:, ix, x0, y0, z1]*(1-x_)*(1-y_)*z_ +
               img[:, ix, x1, y0, z1]*x_*(1-y_)*z_ +
               img[:, ix, x0, y1, z1]*(1-x_)*y_*z_ +
               img[:, ix, x1, y1, z0]*x_*y_*(1-z_) +
               img[:, ix, x1, y1, z1]*x_*y_*z_)

        return out

    def stn(self, x, theta):
        # theta must be (Bs, 3, 4) = [R|t]
        #theta = theta.view(-1, 2, 3)

        if self.pad_rotate:
            spat_dim = x.size(-1)
            # Pad by spat_dim on both sides
            pad = nn.ConstantPad3d(spat_dim//2, 0.)
            x = pad(x)

        grid = F.affine_grid(theta, x.size())
        if x.is_cuda:
            grid = grid.to(self.device)

        # TODO: padding operation

        if self.interp_mode == 'bilinear':
            out = F.grid_sample(x, grid, padding_mode='zeros')
        elif self.interp_mode == 'trilinear':
            #out = [ self.interpolate_trilinear(x[j:j+1],
            #                                   (grid[j, :, :, :, 2]*0.5+0.5)*im_sz,
            #                                   (grid[j, :, :, :, 1]*0.5+0.5)*im_sz,
            #                                   (grid[j, :, :, :, 0]*0.5 + 0.5)*im_sz) \
            #        for j in range(x.size(0)) ]
            #out = torch.cat(out, dim=0)
            im_sz = x.size(-1)
            out = self.interpolate_trilinear(x.transpose(0,1),
                                             (grid[:,:,:,:,2]*0.5 + 0.5)*im_sz,
                                             (grid[:,:,:,:,1]*0.5 + 0.5)*im_sz,
                                             (grid[:,:,:,:,0]*0.5 + 0.5)*im_sz)
            out = out.transpose(1, 0).contiguous()
        else:
            raise Exception("this interp mode not implemented")

        return out

    def rotate_I(self, enc):
        zeros_3 = torch.zeros((enc.size(0), 3)).cuda()
        zeros_1 = torch.zeros((enc.size(0), 1)).cuda()

        R = self.t_get_R(zeros_1,
                         zeros_3,
                         zeros_3)
        return self.generator.post_rot(self.stn(enc, R))

    def rotate(self,
               enc,
               theta,
               n,
               t,
               **kwargs):

        R = self.t_get_R(theta, n, t, **kwargs)

        return self.generator.post_rot(self.stn(enc, R))

    def _generate_rotations(self, x_batch, num=5):
        dd = dict()
        pbuf = []
        with torch.no_grad():
            enc, q = self.generator.encode(x_batch)
            h = self.generator.enc2vol(enc)
            rz, ry, rx, tx, ty, tz = self._extract_angles_and_offsets(q)
            for p in np.linspace(
                    self.min_angle,
                    self.max_angle,
                    num=num):
                h_rot = self.rotate(h,
                                    rz[1], ry[1]*0 + p, rx[1],
                                    tx[1], ty[1], tz[1])
                dec_enc_rot = self.generator.decode(h_rot)
                pbuf.append(dec_enc_rot)
        dd['rot'] = pbuf

        return dd

    def t_get_R(self, theta, vector, offsets=None):
        '''Construct a rotation matrix from angles. (This is
        the differentiable version, in PyTorch code.)

        angles should be an nx3 matrix with z,y,z = 0,1,2
        '''

        n_z = vector[:, 0]
        n_y = vector[:, 1]
        n_x = vector[:, 2]
        if self.fix_n is not None:
            n_z = n_z*0. + self.fix_n[0]
            n_y = n_y*0. + self.fix_n[1]
            n_x = n_x*0. + self.fix_n[2]

        if offsets is not None:
            trans_x = offsets[:, 0]
            trans_y = offsets[:, 1]
            trans_z = offsets[:, 2]
            if self.fix_t is not None:
                trans_x = trans_x*0. + self.fix_t[0]
                trans_y = trans_y*0. + self.fix_t[1]
                trans_z = trans_z*0. + self.fix_t[2]

        N = torch.zeros((vector.size(0), 3, 3)).cuda()

        N[:, 0, 0] = 0.
        N[:, 0, 1] = -n_z
        N[:, 0, 2] = n_y

        N[:, 1, 0] = n_z
        N[:, 1, 1] = 0.
        N[:, 1, 2] = -n_x

        N[:, 2, 0] = -n_y
        N[:, 2, 1] = n_x
        N[:, 2, 2] = 0.

        I_mat = torch.eye(3, 3).unsqueeze(0).\
            repeat(vector.size(0), 1, 1).cuda()

        theta_rshp = theta.unsqueeze(-1).repeat(1, 3, 3)

        R = I_mat + \
            N*(torch.sin(theta_rshp)) + \
            torch.bmm(N, N)*(1. - torch.cos(theta_rshp))

        trans = torch.zeros((vector.size(0), 3, 1))

        if offsets is not None:
            trans[:, 0, :] = trans_x.view(-1, 1)
            trans[:, 1, :] = trans_y.view(-1, 1)
            trans[:, 2, :] = trans_z.view(-1, 1)
        if vector.is_cuda:
            trans = trans.cuda()

        R = torch.cat((R, trans), dim=2) # add zero padding
        return R

    def uniform(self, a, b):
        if self.beta > 0:
            distn = Uniform(a, b)
            return distn, distn.rsample()
        else:
            distn = Uniform(a, a+1e-3)
            return distn, a

    def _extract_angles_and_offsets(self, x):

        def _std(x):
            if self.beta > 0:
                return torch.exp(x)
            else:
                return 0.

        ns_mu_and_logvar = x[:, 0:6]
        ts_mu_and_logvar = x[:, 6:12]
        theta_mu_and_logvar = x[:, 12:]

        ns_a = ns_mu_and_logvar[:, 0:3]
        ns_b = _std(ns_mu_and_logvar[:, 3:6])

        ts_a = ts_mu_and_logvar[:, 0:3]
        ts_b = _std(ts_mu_and_logvar[:, 3:6])

        theta_a = theta_mu_and_logvar[:, 0:1]
        theta_b = _std(theta_mu_and_logvar[:, 1:2])

        # Make n/t point distns for now.
        n = Normal(
            ns_a,
            ns_b*0 + 1e-5
        )
        t = Normal(
            ts_a,
            ts_b*0 + 1e-5
        )
        eps = 1e-2
        theta = Uniform(
            torch.clamp(theta_a, min=-np.pi+eps, max=np.pi-eps),
            torch.clamp(theta_a+theta_b, min=-np.pi+eps, max=np.pi-eps)
        )

        dd = {
            'n': n,
            't': t,
            'theta': theta
        }
        return dd

    def _normal_like(self, enc):
        return torch.zeros_like(enc).normal_(0., 1.)

    def inverse(self, theta, nz, ny, nx, tx, ty, tz):
        R = self.t_get_R(theta,
                         vector=torch.cat((nz, ny, nx), dim=1),
                         offsets=torch.cat((tx, ty, tz), dim=1))
        # Now pad bottom
        zeros = torch.zeros((theta.size(0), 1, 4)).cuda()
        zeros[:, :, -1] = 1.
        R = torch.cat((R, zeros), dim=1)

        inv_R = torch.inverse(R)[:,0:3]

        return inv_R

    def encode(self, x_batch, x2_batch=None):
        if x2_batch is None:
            x2_batch = x_batch
        enc, q = self.generator.encode(x_batch, x2_batch)
        _q = self._extract_angles_and_offsets(q)
        return enc, _q

    def permute_dims(self, z):
        # TODO: add link
        assert z.dim() == 2
        B, _ = z.size()
        perm_z = []
        for z_j in z.split(1, 1):
            perm = torch.randperm(B).to(z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)
        return torch.cat(perm_z, 1)

    def run_on_instance(self,
                        x_batch,
                        x2_batch,
                        q_batch,
                        cam_batch,
                        cam2_batch,
                        y_batch,
                        meta_batch,
                        train=True,
                        **kwargs):
        if train:
            for key in self.optim:
                self.optim[key].zero_grad()

        if self.sigma > 0:
            # stack batches
            x_s_batch = torch.cat((x_batch, x2_batch), dim=0)
            x2_s_batch = torch.cat((x2_batch, x_batch), dim=0)

            x_batch = x_s_batch
            x2_batch = x2_s_batch


        # (1) Reconstruction
        # (3) Supervised reconstruction
        # (4) Inverse loss
        # (5) (Optional) KL

        # -------------------
        # (1) Reconstruction.
        # -------------------
        if not self.disable_multiview:
            enc, q = self.generator.encode(x_batch,
                                           x2_batch)
        else:
            enc, q = self.generator.encode(x_batch,
                                           torch.zeros_like(x_batch))

        _q = self._extract_angles_and_offsets(q)
        n = _q['n']
        t = _q['t']
        theta = _q['theta']
        # Given (x1,x2), we obtain h1, and to
        # reconstruct all we have to do is
        # 'rotate' h1 by I, and decode.
        h1 = self.generator.enc2vol(enc)
        dec_enc = self.generator.decode(self.rotate_I(h1)) # rotate by I here
        recon_loss = torch.mean(torch.abs(dec_enc-x_batch))

        with torch.no_grad():
            theta_norm = (n.rsample()**2).mean() + \
                (t.rsample()**2).mean() + \
                (theta.rsample()**2).mean()

        if not self.disable_multiview:
            if self.blanking_loss:
                # Given the same pair (x1,x1), we obtain h1,
                # rotate by I, and the resulting hb should
                # be the same as the h for the recon loss.
                enc_b, _ = self.generator.encode(x_batch,
                                                 x_batch)
                hb = self.generator.enc2vol(enc_b)
                hb = self.rotate_I(hb)
                h_loss_blank = torch.mean(torch.abs(hb-h1))

        # -------------------------
        # Supervised reconstruction
        # -------------------------

        if not self.disable_multiview:

            # Given (x1,x2), we get R_12 and h1.
            # To get x2, we do h2 = rot(h1, R_12)
            # and decode.
            h1_onto_h2 = self.rotate(h1,
                                     theta=theta.rsample(),
                                     nz=nz,
                                     ny=ny,
                                     nx=nx,
                                     tx=tx,
                                     ty=ty,
                                     tz=tz)
            x_h1_onto_h2 = self.generator.decode(h1_onto_h2)
            recon_sup1 = torch.mean(torch.abs(x2_batch-x_h1_onto_h2))

        # TODO
        if self.sigma > 0:
            raise NotImplementedError()

        if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
            print("Debugging info:")
            print("  x shape:", x_batch.shape)
            print("  h shape:", h1.shape)
            print("  enc shape:", enc.shape)
            print("  recon shape:", dec_enc.shape)
            print("  disable dh:", self.disable_dh)
            print("  projection mode:", self.projection)

        # Fool D with reconstruction.
        if not self.disable_gan:
            disc_g_recon_loss = self.gen_loss(dec_enc)

        if not self.disable_dh:
            d_g_out = self.disc_enc(
                torch.cat((theta.rsample(), n.rsample(), t.rsample()), dim=1)
            )
            d_g_theta = self.bce(d_g_out, 1)

        # ------------------------
        # KL divergence on angles.
        # ------------------------
        if self.beta > 0:
            zeros = torch.zeros((enc.size(0), 1)).float().cuda()
            prior = Uniform(zeros-np.pi, zeros+np.pi)
            kl_loss = kl_divergence(theta, prior).mean()
            #kl_loss = #kl_divergence(t, prior).mean() + \
                #kl_divergence(n, prior).mean() + \
                #kl_divergence(theta, prior).mean()

        if self.disable_multiview:
            gen_loss = self.lamb*recon_loss
        else:
            gen_loss = self.lamb*(recon_loss+recon_sup1)
            if self.blanking_loss:
                gen_loss = gen_loss + h_loss_blank
        if self.beta > 0:
            gen_loss = gen_loss + self.beta*kl_loss

        #if self.sigma > 0:
        #    gen_loss = gen_loss + self.sigma*inv_loss

        if (kwargs['iter']-1) % self.update_g_every == 0:
            if not self.disable_gan:
                gen_loss = gen_loss + self.eps*disc_g_recon_loss #+ disc_g_delta_loss

        if train:
            gen_loss.backward()
            self.optim['g'].step()

        ## ----------------------
        ## Discriminator on image
        ## ----------------------
        if not self.disable_gan:
            self.optim['disc_x'].zero_grad()
            d_losses = []
            # Do reconstruction.
            d_x_fake = self.dis_loss(x_batch, dec_enc.detach())
            d_losses.append(d_x_fake)
            #d_x_fake_delta = self.dis_loss(x_batch, dec_enc_delta.detach())
            #d_losses.append(d_x_fake_delta)
            d_x = sum(d_losses)
            d_x.backward()
            self.optim['disc_x'].step()

        if not self.disable_dh:
            if train:
                self.optim['disc_enc'].zero_grad()
            theta_joint = torch.cat(
                (theta.rsample(), n.rsample(), t.rsample()), dim=1)
            theta_marginal = self.permute_dims(theta_joint)
            # "real" = marginal, "joint" = "fake"
            d_joint = self.disc_enc(theta_joint.detach())
            d_marginal = self.disc_enc(theta_marginal.detach())
            d_theta = self.bce(d_joint, 0) + self.bce(d_marginal, 1)
            if train:
                d_theta.backward()
                self.optim['disc_enc'].step()

        ## -------------------
        ## Probe, if it exists
        ## -------------------

        if self.probe is not None:
            self.optim['probe'].zero_grad()

            if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
                print("Debugging info:")
                print("  h_rot shape:", h_rot.shape)

            # In this class, we pass the volume
            # to the probe.
            probe_out = self.probe(h_rot.detach(), q_batch, cam_batch)
            probe_loss = self.cls_loss_fn(probe_out, y_batch)

            if train:
                probe_loss.backward()
                self.optim['probe'].step()
            with torch.no_grad():
                if self.cls_loss_fn != self.mse:
                    probe_acc = (probe_out.argmax(dim=1).long() == y_batch).float().mean()
                else:
                    probe_acc = probe_loss

        with torch.no_grad():
            theta_sample = theta.rsample()
        losses = {
            'gen_loss': gen_loss.item(),
            'recon': recon_loss.item(),
            'recon_sup': (recon_sup1).item() / 1.,
            'theta_min': theta_sample.min().item(),
            'theta_max': theta_sample.max().item(),
            'theta_std': theta_sample.std().item()
            #'theta_norm': theta_norm.item()
        }
        if self.beta > 0:
            losses['kl_loss'] = kl_loss.item()

        if not self.disable_multiview:
            losses['recon_sup'] = recon_sup1.item()
            if self.blanking_loss:
                losses['h_blank'] = h_loss_blank.item()

        if not self.disable_gan:
            losses['disc_g_recon'] = disc_g_recon_loss.item()
            #losses['disc_g_delta'] = disc_g_delta_loss.item()
            losses['d_x'] = d_x.item() / len(d_losses)

        if not self.disable_dh:
            losses['d_theta'] = d_theta.item()
            losses['d_g_theta'] = d_g_theta.item()

        if self.probe is not None:
            losses['probe_loss'] = probe_loss.item()
            losses['probe_acc'] = probe_acc.item()

        outputs = {
            'recon': dec_enc,
            'input': x_batch,
            'input2': x2_batch,
            'x_undo': dec_enc*0.
        }
        if self.disable_multiview:
            outputs['x_h1_onto_h2'] = dec_enc*0.
        else:
            outputs['x_h1_onto_h2'] = x_h1_onto_h2

        return losses, outputs

    def train_on_instance(self,
                          x_batch,
                          x2_batch,
                          q_batch,
                          cam_batch,
                          cam2_batch,
                          y_batch,
                          meta_batch,
                          **kwargs):
        self._train()
        return self.run_on_instance(
            x_batch,
            x2_batch,
            q_batch,
            cam_batch,
            cam2_batch,
            y_batch,
            meta_batch,
            True,
            **kwargs
        )

    def eval_on_instance(self,
                         x_batch,
                         x2_batch,
                         q_batch,
                         cam_batch,
                         cam2_batch,
                         y_batch,
                         meta_batch,
                         **kwargs):
        self._eval()
        with torch.no_grad():
            return self.run_on_instance(
                x_batch,
                x2_batch,
                q_batch,
                cam_batch,
                cam2_batch,
                y_batch,
                meta_batch,
                False,
                **kwargs
            )

    def save(self, filename, epoch):
        dd = {}
        # Save the models.
        dd['g'] = self.generator.state_dict()
        if self.disc_x is not None:
            dd['disc_x'] = self.disc_x.state_dict()
        if self.disc_enc is not None:
            dd['disc_enc'] = self.disc_enc.state_dict()
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
        if self.disc_x is not None:
            self.disc_x.load_state_dict(dd['disc_x'], strict=self.load_strict)
        if self.disc_enc is not None:
            self.disc_enc.load_state_dict(dd['disc_enc'], strict=self.load_strict)
        if self.probe is not None:
            self.probe.load_state_dict(dd['probe'], strict=self.load_strict)
        # Load the models' optim state.
        for key in self.optim:
            if ('optim_%s' % key) in dd:
                self.optim[key].load_state_dict(dd['optim_%s' % key])
        self.last_epoch = dd['epoch']
