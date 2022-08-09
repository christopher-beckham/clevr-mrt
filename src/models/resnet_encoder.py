import torch
from collections import OrderedDict
from torch import optim
from torch.nn import functional as F
from itertools import chain
from .base import Base
from torch import nn
import numpy as np

from .. import setup_logger
logger = setup_logger.get_logger()

class ResnetEncoder(Base):
    """
    Intended to be used as a FILM baseline. This does not
    support explicit rotation of volumes via cam2rotation.
    """

    def __init__(self,
                 enc,
                 probe,
                 cls_loss='cce',
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002,
                           'betas': (0.5, 0.999)},
                 scheduler=None,
                 handlers=[]):
        super(ResnetEncoder, self).__init__()

        use_cuda = True if torch.cuda.is_available() else False
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.enc = enc
        self.probe = probe

        self.use_cuda = use_cuda
        if self.use_cuda:
            if self.probe is not None:
                self.probe.to(self.device)

        if cls_loss == 'cce':
            self.cls_loss_fn = nn.CrossEntropyLoss()
        elif cls_loss == 'mse':
            self.cls_loss_fn = self.mse
        else:
            raise Exception("Only cce or mse is currently supported for cls_loss")

        self._optim = {}
        optim_probe = opt(filter(lambda p: p.requires_grad, probe.parameters()), **opt_args)
        self._optim['probe'] = optim_probe

        self._scheduler = scheduler
        self._handlers = handlers

        self.last_epoch = 0
        self.load_strict = True

        self._metric_highest = -np.inf

    def _train(self):
        # Encoder is not trained, it is always
        # in eval mode.
        self.enc.eval()
        if self.probe is not None:
            self.probe.train()

    def _eval(self):
        self.enc.eval()
        if self.probe is not None:
            self.probe.eval()

    def mse(self, prediction, target):
        return torch.mean((prediction-target)**2)

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
        for key in self.optim:
            self.optim[key].zero_grad()

        if hasattr(self.enc, 'encode'):
            enc = self.enc.encode(x_batch)[0].detach()
        else:
            # Must be imagenet model
            enc = self.enc(x_batch).detach()

        if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
            logger.info("x shape: {}".format(x_batch.shape))
            logger.info("enc shape: {}".format(enc.shape))
            logger.info("cam shape: {}".format(cam_batch.shape))

        probe_out = self.probe(enc, q_batch, cam_batch)
        probe_loss = self.cls_loss_fn(probe_out, y_batch)
        probe_loss.backward()
        with torch.no_grad():
            if self.cls_loss_fn != self.mse:
                probe_acc = (probe_out.argmax(dim=1).long() == y_batch).float().mean()
            else:
                probe_acc = probe_loss
        self.optim['probe'].step()

        losses = {}
        losses['probe_loss'] = probe_loss.item()
        losses['probe_acc'] = probe_acc.item()

        outputs = {}

        return losses, outputs

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
        losses = {}
        with torch.no_grad():

            if hasattr(self.enc, 'encode'):
                enc = self.enc.encode(x_batch)[0].detach()
            else:
                # Must be imagenet model
                enc = self.enc(x_batch).detach()

            probe_out = self.probe(enc, q_batch, cam_batch)
            probe_loss = self.cls_loss_fn(probe_out, y_batch)
            if self.cls_loss_fn != self.mse:
                probe_acc = (probe_out.argmax(dim=1).long() == y_batch).float().mean()
            else:
                probe_acc = probe_loss
            losses['probe_loss'] = probe_loss.item()
            losses['probe_acc'] = probe_acc.item()

        return losses, {}

    def predict(self,
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
            if hasattr(self.enc, 'encode'):
                enc = self.enc.encode(x_batch)[0].detach()
            else:
                # Must be imagenet model
                enc = self.enc(x_batch).detach()

            probe_out = self.probe(enc, q_batch, cam_batch)
            return probe_out

    def save(self, filename, epoch):
        dd = {}
        # Save the models.
        dd['probe'] = self.probe.state_dict()
        # Save the models' optim state.
        for key in self.optim:
            dd['optim_%s' % key] = self.optim[key].state_dict()
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
        self.last_epoch = dd['epoch']
        if 'metric_lowest' in dd:
            self.set_metric_highest(dd['metric_highest'])
        else:
            logger.warning("Did not find `metric_lowest` in chkpt")