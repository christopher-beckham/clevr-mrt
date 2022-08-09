import torch
import numpy as np
from collections import OrderedDict
from torch import optim
from itertools import chain
from .base import Base
from torch import nn

class Classifier(Base):

    def __init__(self,
                 cls,
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 handlers=[]):

        super(Classifier, self).__init__()

        use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        if len(batch) != 3:
            raise Exception("Expected batch to only contain X, z, and y")
        X_batch = batch[0].float()
        z_batch = batch[1].float()
        y_batch = batch[2].long()
        if self.use_cuda:
            X_batch = X_batch.cuda()
            z_batch = z_batch.cuda()
            y_batch = y_batch.cuda()
        return [X_batch, z_batch, y_batch]

    def _top_k(self, preds, y, k):
        y_k = y.view(-1, 1).repeat(1, k)
        preds_k = preds.topk(k)[1]
        acc_k = (y_k == preds_k).any(dim=1).float().mean()
        return acc_k

    def train_on_instance(self,
                          x_batch,
                          z_batch,
                          y_batch,
                          **kwargs):
        self._train()
        self.optim['cls'].zero_grad()

        cls_out = self.cls(x_batch, z_batch)

        cls_preds_log = torch.log_softmax(cls_out, dim=1)
        cls_loss = nn.NLLLoss()(cls_preds_log,
                                y_batch)

        with torch.no_grad():
            cls_preds = torch.softmax(cls_out, dim=1)
            cls_acc = (cls_preds.argmax(dim=1) == y_batch).float().mean()

            cls_acc3 = self._top_k(cls_preds, y_batch, 3)
            cls_acc5 = self._top_k(cls_preds, y_batch, 5)

        cls_loss.backward()
        self.optim['cls'].step()

        losses = {}
        losses['loss'] = cls_loss.item()
        losses['acc'] = cls_acc.item()
        losses['acc3'] = cls_acc3.item()
        losses['acc5'] = cls_acc5.item()

        outputs = {
        }

        return losses, outputs

    def eval_on_instance(self,
                         x_batch,
                         z_batch,
                         y_batch,
                         **kwargs):
        self._eval()
        with torch.no_grad():

            cls_out = self.cls(x_batch, z_batch)
            cls_preds = torch.softmax(cls_out, dim=1)

            cls_acc = (cls_preds.argmax(dim=1) == y_batch).float().mean()

            cls_acc3 = self._top_k(cls_preds, y_batch, 3)
            cls_acc5 = self._top_k(cls_preds, y_batch, 5)

            losses = {}
            losses['acc'] = cls_acc.item()
            losses['acc3'] = cls_acc3.item()
            losses['acc5'] = cls_acc5.item()
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
