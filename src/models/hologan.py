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

class HoloGAN(Base):

    def __init__(self,
                 generator,
                 disc_x,
                 lamb=1.0,
                 gamma=0.0,
                 z_dim=128,
                 interp_mode='bilinear',
                 fix_t=None,
                 fix_n=None,
                 gan_loss='bce',
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 update_g_every=1,
                 handlers=[]):
        super(HoloGAN, self).__init__()

        use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.generator = generator
        self.disc_x = disc_x
        self.interp_mode = interp_mode
        self.fix_n = fix_n
        self.fix_t = fix_t
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.lamb = lamb
        self.gamma = gamma

        if self.use_cuda:
            self.generator.to(self.device)
            self.disc_x.to(self.device)

        optim_g = opt(filter(lambda p: p.requires_grad,
                             self.generator.parameters()), **opt_args)
        optim_disc_x = opt(filter(lambda p: p.requires_grad,
                                  self.disc_x.parameters()), **opt_args)
        self.optim = {'g': optim_g, 'disc_x': optim_disc_x}

        self.schedulers = []

        self.update_g_every = update_g_every
        self.handlers = handlers

        self.pad_rotate = False

        ##################
        # Loss functions #
        ##################
        if gan_loss == 'bce':
            self.gan_loss_fn = self.bce
        elif gan_loss == 'mse':
            self.gan_loss_fn = self.mse
        else:
            raise Exception("Only bce or mse is currently supported for gan_loss")

        self.last_epoch = 0
        self.load_strict = True

    def train(self):
        self._train()

    def eval(self):
        self._eval()

    def _train(self):
        self.generator.train()
        self.disc_x.train()

    def _eval(self):
        self.generator.eval()
        self.disc_x.eval()

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

    def sample(self, x_batch):
        """Output a random sample (a rotation)
        :param x_batch: the batch to randomly rotate.
        :returns:
        :rtype:
        """
        raise NotImplementedError()

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
            im_sz = x.size(-1)
            out = self.interpolate_trilinear(x.transpose(0,1),
                                             (grid[:,:,:,:,2]*0.5 + 0.5)*im_sz,
                                             (grid[:,:,:,:,1]*0.5 + 0.5)*im_sz,
                                             (grid[:,:,:,:,0]*0.5 + 0.5)*im_sz)
            out = out.transpose(1, 0).contiguous()
        else:
            raise Exception("this interp mode not implemented")

        return out

    def rotate(self,
               enc,
               theta,
               n,
               t,
               **kwargs):

        R = self.t_get_R(theta, n, t, **kwargs)

        return self.generator.post_rot(self.stn(enc, R))

    def _generate_rotations(self, bs, num=20):
        pbuf = []
        with torch.no_grad():
            z = self.sample_z(bs).cuda()
            zeros = self.sample_theta(bs).cuda()*0.
            h = self.generator.enc2vol(z)
            for p in np.linspace(
                    -np.pi,
                    np.pi,
                    num=num):
                h_rot = self.rotate(h,
                                    theta=zeros+p,
                                    n=torch.zeros((bs, 3)).cuda(),
                                    t=torch.zeros((bs, 3)).cuda())
                dec_enc_rot = self.generator.decode(h_rot)
                pbuf.append(dec_enc_rot)
        return pbuf

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

    def sample_z(self, bs):
        n = Normal(torch.zeros(bs, self.z_dim),
                   torch.ones(bs, self.z_dim))
        return n.sample().detach()

    def sample_theta(self, bs):
        # Define range to be from -180 deg to +180,
        # i.e. 360 deg overall.
        n = Uniform(torch.zeros(bs, 1)-np.pi,
                    torch.zeros(bs, 1)+np.pi)
        return n.sample().detach()

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

        z = self.sample_z(x_batch.size(0)).cuda()
        theta = self.sample_theta(x_batch.size(0)).cuda()

        h = self.generator.enc2vol(z)
        bs = x_batch.size(0)
        h_rot = self.rotate(h,
                            theta=theta,
                            n=torch.zeros((bs, 3)).cuda(),
                            t=torch.zeros((bs, 3)).cuda())
        x_fake = self.generator.decode(h_rot)

        d_g_out, d_g_z_out, d_g_t_out = self.disc_x(x_fake)

        d_g_loss = self.bce(d_g_out, 1)
        z_g_loss = torch.mean((d_g_z_out-z)**2)
        t_g_loss = torch.mean((d_g_t_out-theta)**2)

        if train:
            if (kwargs['iter']-1) % self.update_g_every == 0:
                (d_g_loss + self.lamb*(z_g_loss) + t_g_loss).backward()
                self.optim['g'].step()

        if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
            print("Debugging info:")
            print("  x_fake:", x_fake.shape)
            print("  x_real:", x_batch.shape)
            print("  d_out:", d_g_out.shape)

        if train:
            self.optim['disc_x'].zero_grad()

        d_real, d_zr_out, d_tr_out = self.disc_x(x_batch)
        d_fake, d_z_out, d_t_out = self.disc_x(x_fake.detach())
        d_loss = self.bce(d_real, 1) + self.bce(d_fake, 0)

        z_loss = torch.mean((d_z_out-z)**2)
        t_loss = torch.mean((d_t_out-theta)**2)

        if train:
            tot_loss = d_loss + self.lamb*(z_loss) + t_loss
            tot_loss.backward()
            self.optim['disc_x'].step()

        self.optim['disc_x'].zero_grad()
        self.optim['g'].zero_grad()

        if self.gamma > 0:
            # Find z|x and theta|x
            _, d_zr_out, d_tr_out = self.disc_x(x_batch)
            # Run it through the generator and decode
            h_x = self.generator.enc2vol(d_zr_out)
            h_x_rot = self.rotate(h_x,
                                  theta=d_tr_out,
                                  n=torch.zeros((bs, 3)).cuda(),
                                  t=torch.zeros((bs, 3)).cuda())
            x_recon = self.generator.decode(h_x_rot)
            dg_recon_loss = torch.mean(torch.abs(x_recon-x_batch))
            if train:
                # min recon loss wrt both sets of params
                dg_recon_loss.backward()
                self.optim['disc_x'].step()
                self.optim['g'].step()
        else:
            x_recon = x_batch*0.

        losses = {
            'd_g_loss': d_g_loss.item(),
            'd_loss': d_loss.item() / 2.,
            'z_g_loss': z_g_loss.item(),
            'z_loss': z_loss.item(),
            't_g_loss': t_g_loss.item(),
            't_loss': t_loss.item()
        }
        if self.gamma > 0:
            losses['dg_recon_loss'] = dg_recon_loss.item()

        outputs = {}
        if kwargs['iter'] == 1:
            outputs = {
                'x_fake': x_fake,
                'rot': self._generate_rotations(x_batch.size(0)),
                'x_recon': x_recon,
                'x_real': x_batch
            }

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
        dd['disc_x'] = self.disc_x.state_dict()
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
        self.disc_x.load_state_dict(dd['disc_x'], strict=self.load_strict)
        # Load the models' optim state.
        for key in self.optim:
            if ('optim_%s' % key) in dd:
                self.optim[key].load_state_dict(dd['optim_%s' % key])
        self.last_epoch = dd['epoch']
