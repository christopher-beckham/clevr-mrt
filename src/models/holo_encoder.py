import numpy as np
import torch
from collections import OrderedDict
from torch import optim
from torch.nn import functional as F
from itertools import chain
from .base import Base
from .holo_ae import HoloAE
from torch import nn

from .util.rotation import (t_get_relative_transform,
                            t_get_theta,
                            t_rot_matrix_x,
                            t_rot_matrix_y,
                            t_rot_matrix_z)

from .. import setup_logger
logger = setup_logger.get_logger()

class HoloEncoder(Base):

    def __init__(self,
                 enc,
                 probe,
                 disable_rot=False,
                 supervised=False,
                 cls_loss='cce',
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002,
                           'betas': (0.5, 0.999)},
                 scheduler=None,
                 handlers=[],
                 ignore_last_epoch=False):
        super(HoloEncoder, self).__init__()

        use_cuda = True if torch.cuda.is_available() else False
        self._device = torch.device("cuda:0" if use_cuda else "cpu")

        self.enc = enc
        self.probe = probe
        self.disable_rot = disable_rot
        self.supervised = supervised
        self.ignore_last_epoch = ignore_last_epoch

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.enc.to(self.device)
            if self.probe is not None:
                self.probe.to(self.device)

        if cls_loss == 'cce':
            self.cls_loss_fn = nn.CrossEntropyLoss()
        elif cls_loss == 'mse':
            self.cls_loss_fn = self.mse
        else:
            raise Exception("Only cce or mse is currently supported for cls_loss")

        self._optim = {
            'g': opt(
                filter(lambda p: p.requires_grad, enc.parameters()),
            **opt_args)
        }

        self.opt_class = opt
        self.opt_args = opt_args

        if self.probe is not None:
            optim_probe = opt(
                filter(lambda p: p.requires_grad, probe.parameters()),
                **opt_args)
            self._optim['probe'] = optim_probe

        for elem in self.optim.values():
            print(elem)

        self._scheduler = scheduler
        self._handlers = handlers

        self.last_epoch = 0
        self.load_strict = True
        self._metric_highest = -np.inf

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def handlers(self):
        return self._handlers

    @property
    def optim(self):
        return self._optim

    @property
    def schedulers(self):
        return self._schedulers

    @property
    def device(self):
        return self._device
    
    def set_metric_highest(self, val):
        self._metric_highest = val
    
    def get_metric_highest(self):
        return self._metric_highest

    def _train(self):
        self.enc.train()
        if self.probe is not None:
            self.probe.train()

    def train(self):
        self._train()

    def _eval(self):
        self.enc.eval()
        if self.probe is not None:
            self.probe.eval()

    def eval(self):
        self._eval()

    def mse(self, prediction, target):
        return torch.mean((prediction-target)**2)

    def stn(self, x, theta):
        pad = nn.ConstantPad3d(4, 0.)
        x = pad(x)

        grid = F.affine_grid(theta, x.size())
        if x.is_cuda:
            grid = grid.cuda()

        # TODO: padding operation

        out = F.grid_sample(x, grid, padding_mode='zeros')

        out = out[:,:,4:-4,4:-4,4:-4]

        return out

    def _pad(self, x):
        zeros = torch.zeros((x.size(0), 1, 4)).cuda()
        zeros[:, :, -1] = 1.
        x_pad = torch.cat((x, zeros), dim=1)
        return x_pad

    def run_on_instance(self,
                        x_batch,
                        x2_batch,
                        q_batch,
                        cam_batch,
                        cam2_batch,
                        y_batch,
                        cc_batch,
                        meta_batch,
                        train=True,
                        **kwargs):
        if train:
            for key in self.optim:
                self.optim[key].zero_grad()

        losses = {}

        z = self.enc.encode(x_batch)
        h = self.enc.enc2vol(z) # 'template canonical'

        if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
            logger.info("x shape: {}".format(x_batch.shape))
            logger.info("cam shape: {}".format(cam_batch.shape))
            logger.info("z shape: {}".format(z.shape))
            logger.info("h shape: {}".format(h.shape))
            logger.info("is supervised camera: {}".format(self.supervised))

        if not self.disable_rot:
            if not self.supervised:
                # If we're not in supervised mode, we
                # basically take the coords of the viewpoint
                # camera, embed them, and use that embedding
                # to construct a rotation matrix to rotate h.
                theta = self.enc.cam_encode(cam_batch)
                rot_mat = t_get_theta(theta[:, 0:3], theta[:, 3:6])
                h_rot = self.stn(h, rot_mat) # actual viewpoint
            else:
                # If we're in supervised mode, we use `cc_batch`
                # (the canonical camera) to construct a rot
                # matrix that maps from the canonical viewpoint
                # to the viewpoint camera, and use that to
                # rotate h directly.
                rot_mat = t_get_relative_transform(
                    cc_batch, cam_batch, t_lambda=0.1
                )
                                
                h_rot = self.stn(h, rot_mat)
                with torch.no_grad():
                    losses['h_0s'] = (h == 0).float().mean().item()
                    losses['h_rot_0s'] = (h_rot == 0).float().mean().item()
        else:
            h_rot = h

        #z = self.enc.encode(x_batch)
        #h_rot = z

        if self.probe is not None:

            probe_out = self.probe(h_rot, q_batch, cam_batch)
            probe_loss = self.cls_loss_fn(probe_out, y_batch)

            if train:
                probe_loss.backward()
                self.optim['g'].step()
                self.optim['probe'].step()

            with torch.no_grad():
                if self.cls_loss_fn != self.mse:
                    probe_acc = (probe_out.argmax(dim=1).long() == y_batch).float().mean()
                else:
                    probe_acc = probe_loss

            losses['probe_loss'] = probe_loss.item()
            losses['probe_acc'] = probe_acc.item()

        return losses, {}

    def train_on_instance(self,
                          x_batch,
                          x2_batch,
                          q_batch,
                          cam_batch,
                          cam2_batch,
                          y_batch,
                          cc_batch,
                          meta_batch,
                          **kwargs):
        self._train()
        return self.run_on_instance(x_batch,
                                    x2_batch,
                                    q_batch,
                                    cam_batch,
                                    cam2_batch,
                                    y_batch,
                                    cc_batch,
                                    meta_batch,
                                    True,
                                    **kwargs)

    def eval_on_instance(self,
                         x_batch,
                         x2_batch,
                         q_batch,
                         cam_batch,
                         cam2_batch,
                         y_batch,
                         cc_batch,
                         meta_batch,
                         **kwargs):
        self._eval()
        with torch.no_grad():
            return self.run_on_instance(x_batch,
                                        x2_batch,
                                        q_batch,
                                        cam_batch,
                                        cam2_batch,
                                        y_batch,
                                        cc_batch,
                                        meta_batch,
                                        False,
                                        **kwargs)

    def predict(self,
                x_batch,
                x2_batch,
                q_batch,
                cam_batch,
                cam2_batch,
                y_batch=None,
                cc_batch=None,
                meta_batch=None,
                **kwargs):
        self._eval()
        with torch.no_grad():
            z = self.enc.encode(x_batch)
            h = self.enc.enc2vol(z) # 'template canonical'
            if not self.disable_rot:
                theta = self.enc.cam_encode(cam_batch)
                rot_mat = t_get_theta(theta[:, 0:3], theta[:, 3:6])
                h_rot = self.stn(h, rot_mat) # actual viewpoint
            else:
                h_rot = h

            probe_out = self.probe(h_rot, q_batch, cam_batch)
            return probe_out

    def save(self, filename, epoch):
        dd = {}
        # Save the models.
        dd['enc'] = self.enc.state_dict()
        if self.probe is not None:
            dd['probe'] = self.probe.state_dict()
        # Save the models' optim state.
        for key in self._optim:
            dd['optim_%s' % key] = self._optim[key].state_dict()
        dd['epoch'] = epoch
        dd['metric_lowest'] = self.get_metric_highest()
        torch.save(dd, filename)

    def load(self, filename):
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        dd = torch.load(filename,
                        map_location=map_location)

        # Load the models.
        self.enc.load_state_dict(dd['enc'],
                                 strict=self.load_strict)
        if self.probe is not None:
            # HACKY: due to some stupid decisions I made, probe class has a
            # layer called "cam_encode_3d" which is only used for contrastive
            # stuff. So if the checkpoint does not contain keys that start with
            # "cam_encode_3d", then set self.probe.cam_encode_3d to None and
            # then load in the checkpoint.
            any_keys_cam3d = list(filter(lambda st: st.startswith("cam_encode_3d"), 
                                        dd['probe'].keys()))
            if len(any_keys_cam3d) == 0:
                logger.debug("set self.probe.cam_encode_3d=None (see comment in file)...")
                self.probe.cam_encode_3d = None
            self.probe.load_state_dict(dd['probe'], strict=self.load_strict)
        # Load the models' optim state.
        try:
            for key in self.optim:
                if ('optim_%s' % key) in dd:
                    self.optim[key].load_state_dict(dd['optim_%s' % key])
        except:
            print("WARNING: was unable to load state dict for optim")
            print("This is not a big deal if you're only using " + \
                  "the model for inference, however.")
        if not self.ignore_last_epoch:
            self.last_epoch = dd['epoch']
        else:
            print("Last epoch indicated in chkpt is %s but since " + \
                "ignore_last_epoch==True we ignore it here..." % str(dd['epoch']))

        if 'metric_lowest' in dd:
            self.set_metric_highest(dd['metric_highest'])
        else:
            logger.warning("Did not find `metric_lowest` in chkpt")