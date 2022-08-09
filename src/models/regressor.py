import torch
import numpy as np
from torch import optim
from .base import Base
from torch import nn

class Regressor(Base):

    def __init__(self,
                 cls,
                 sigma=0.05,
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 handlers=[]):

        super(Regressor, self).__init__()

        use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.sigma = sigma
        self.cls = cls
        self.schedulers = []

        optim_cls = opt(filter(lambda p: p.requires_grad, cls.parameters()),
                        **opt_args)
        self.optim = {
            'cls': optim_cls,
        }

        self.handlers = handlers
        self.use_cuda = use_cuda
        ########
        # cuda #
        ########
        if self.use_cuda:
            self.cls.to(self.device)

        self.last_epoch = 0
        self.load_strict = True

    def _train(self):
        self.cls.train()

    def _eval(self):
        self.cls.eval()

    def prepare_batch(self, batch):
        if len(batch) != 2:
            raise Exception("Expected batch to only contain X and y")
        X_batch = batch[0].float()
        y_batch = batch[1].float()
        if self.use_cuda:
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()
        return [X_batch, y_batch]

    def train_on_instance(self,
                          x_batch,
                          y_batch,
                          **kwargs):
        self._train()
        self.optim['cls'].zero_grad()

        noise = torch.zeros_like(x_batch).normal_(0, self.sigma)

        if x_batch.is_cuda:
            noise = noise.cuda()

        cls_out = self.cls(x_batch+noise)

        cls_loss = torch.mean(torch.abs(cls_out-y_batch))

        cls_loss.backward()
        self.optim['cls'].step()

        losses = {}
        losses['loss'] = cls_loss.item()

        outputs = {
        }

        return losses, outputs

    def eval_on_instance(self,
                         x_batch,
                         y_batch,
                         **kwargs):
        self._eval()

        with torch.no_grad():

            cls_out = self.cls(x_batch)
            cls_loss = torch.mean(torch.abs(cls_out-y_batch))

            losses = {'loss': cls_loss.item()}
            outputs = {}

            return losses, outputs

    def save(self, filename, epoch):
        dd = {}
        # Save the models.
        dd['cls'] = self.cls.state_dict()
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
        self.cls.load_state_dict(dd['cls'], strict=self.load_strict)
        # Load the models' optim state.
        for key in self.optim:
            self.optim[key].load_state_dict(dd['optim_%s' % key])
        self.last_epoch = dd['epoch']
