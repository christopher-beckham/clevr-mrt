import torch
from torch import nn
from torch import optim
from .classifier import Classifier
from itertools import chain

'''
>>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
>>> anchor = torch.randn(100, 128, requires_grad=True)
>>> positive = torch.randn(100, 128, requires_grad=True)
>>> negative = torch.randn(100, 128, requires_grad=True)
>>> output = triplet_loss(anchor, positive, negative)
'''

class SiameseExplicitDistanceClassifier(Classifier):
    '''Siamese network with ground truth target distances.

    In other words, rather than classify +ve / -ve pairs, we specify
    the actual distance we want for a particular pair.
    '''

    def __init__(self, sigma, *args, **kwargs):
        super(SiameseExplicitDistanceClassifier, self).__init__(*args, **kwargs)

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

        perm = torch.randperm(x_batch.size(0))
        x_batch_2 = x_batch[perm]
        y_batch_2 = y_batch[perm]

        embed1 = self.cls.embed(x_batch)
        embed2 = self.cls.embed(x_batch_2)

        y_batch_diff = torch.abs(y_batch - y_batch_2)

        pred_dist = torch.sum(torch.abs(embed1-embed2), dim=1)

        loss = torch.mean(torch.abs(pred_dist - y_batch_diff))

        loss.backward()
        self.optim['cls'].step()

        losses = {}
        losses['loss'] = loss.item()

        outputs = {
        }

        return losses, outputs

    def eval_on_instance(self,
                         x_batch,
                         y_batch,
                         **kwargs):
        self._eval()
        with torch.no_grad():
            return {}, {}

class SiameseBalancedSamplingClassifier(Classifier):
    ''' Siamese classifier with 'balanced' sampling.

    That means that each minibatch contains a batch of 'positive'
    (i.e., all instances inside have the same ID) and also a batch
    of 'negative' examples (all instances inside come from different
    IDs).

    TODO: refactor this to use some sort of superclass 

    '''

    def __init__(self, *args, **kwargs):
        self.sigma = kwargs.pop('sigma')
        super(SiameseBalancedSamplingClassifier, self).__init__(*args, **kwargs)
        if not hasattr(self.cls, 'embed') and \
           not hasattr(self.cls, 'out'):
            raise Exception("The given network must have the two methods: " +
                            "`embed` and `out`.")

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
        xn1, xn2 = self._noise(xn_batch[:, 0:3]), self._noise(xn_batch[:, 3:6])

        cls_pos = self.cls(xp1, xp2)
        cls_neg = self.cls(xn1, xn2)

        cls_same = self.cls(xp1, xp1)

        with torch.no_grad():
            acc = ((cls_pos >= 0.5).float().mean() + \
                  (cls_neg < 0.5).float().mean()) / 2.
            same_acc = (cls_same >= 0.5).float().mean()

        loss = self.bce(cls_pos, 1) + self.bce(cls_neg, 0) + \
            self.bce(cls_same, 1)

        loss.backward()
        self.optim['cls'].step()

        losses = {}
        losses['loss'] = loss.item()
        losses['acc'] = acc.item()
        losses['same_acc'] = same_acc.item()

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
            xn1, xn2 = xn_batch[:, 0:3], xn_batch[:, 3:6]

            cls_pos = self.cls(xp1, xp2)
            cls_neg = self.cls(xn1, xn2)

            cls_same = self.cls(xp1, xp1)

            acc = ((cls_pos >= 0.5).float().mean() + \
                  (cls_neg < 0.5).float().mean()) / 2.
            same_acc = (cls_same >= 0.5).float().mean()

            return {
                'acc': acc.item(),
                'same_acc': same_acc.item()
            }, {}

class SiameseTripletMarginClassifier(Classifier):
    '''
    '''

    def __init__(self, *args, **kwargs):
        self.sigma = kwargs.pop('sigma')
        super(SiameseTripletMarginClassifier, self).__init__(*args, **kwargs)
        if not hasattr(self.cls, 'embed') and \
           not hasattr(self.cls, 'out'):
            raise Exception("The given network must have the two methods: " +
                            "`embed` and `out`.")
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

        # Ok, construct a +ve example
        #pos = torch.cat((img1, img2), dim=0)
        # Ok, construct a -ve example
        #neg = torch.cat((img1, img3), dim=0) # or img2+img3


    def train_on_instance(self,
                          xp_batch,
                          xn_batch,
                          **kwargs):
        self._train()
        self.optim['cls'].zero_grad()

        xp1, xp2 = self._noise(xp_batch[:, 0:3]), self._noise(xp_batch[:, 3:6])
        xn = self._noise(xn_batch[:, 3:6])

        embed_xp1 = self.cls.embed(xp1)
        embed_xp2 = self.cls.embed(xp2)
        embed_xn = self.cls.embed(xn)

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

            embed_xp1 = self.cls.embed(xp1)
            embed_xp2 = self.cls.embed(xp2)
            embed_xn = self.cls.embed(xn)

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
