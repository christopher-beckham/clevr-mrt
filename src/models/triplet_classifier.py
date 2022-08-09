import torch
from torch import nn
from .classifier import Classifier

class TripletClassifier(Classifier):
    '''Triplet classifier, made specifically for the IQTT dataset.

    Batches of X are in the format (bs, nc*4, h, w), where the 2nd
    axis contains 4 `nc`-channeled images.
    '''

    def __init__(self, *args, **kwargs):
        super(TripletClassifier, self).__init__(*args, **kwargs)
        self.loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    def prepare_batch(self, batch):
        if len(batch) != 2:
            raise Exception("Expected batch to only contain X and y")
        X_batch = batch[0].float()
        y_batch = batch[1].long()
        if self.use_cuda:
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()
        return [X_batch, y_batch]

    def _embed(self, x_batch):
        x_batch = x_batch.view((-1, 3, x_batch.size(2), x_batch.size(3)))
        z_batch = torch.ones((x_batch.size(0), 3)).cuda()
        embeddings = self.cls(x_batch, z_batch)
        embeddings = embeddings.view((-1, 4, embeddings.size(1)))
        anchor = embeddings[:, 0]
        q1 = embeddings[:, 1]
        q2 = embeddings[:, 2]
        q3 = embeddings[:, 3]
        return anchor, q1, q2, q3

    def predict(self, x_batch):
        self._eval()
        with torch.no_grad():
            anchor, q1, q2, q3 = self._embed(x_batch)
            # Compute the Euclidean distance between the anchor and
            # each of the three queries.
            d_anchor_q1 = torch.sum((anchor-q1)**2, dim=1, keepdim=True)
            d_anchor_q2 = torch.sum((anchor-q2)**2, dim=1, keepdim=True)
            d_anchor_q3 = torch.sum((anchor-q3)**2, dim=1, keepdim=True)
            dd = torch.cat((d_anchor_q1, d_anchor_q2, d_anchor_q3),
                           dim=1)
            preds = dd.min(dim=1)[1]
            return preds

    def train_on_instance(self,
                          x_batch,
                          y_batch,
                          **kwargs):
        self._train()
        self.optim['cls'].zero_grad()

        n = x_batch.size(0)

        # batches come in as (n, 3*4, 64, 64),
        # so reshape them as (n*4, 3, 64, 64)
        x_batch = x_batch.view((-1, 3, x_batch.size(2), x_batch.size(3)))

        # assertion
        #tmp = x_batch_.view((n, 3*4, x_batch.size(2), x_batch.size(3)))
        # assert that tmp == original x_batch

        z_batch = torch.ones((x_batch.size(0), 3)).cuda()

        # (n*4, 3, 64, 64) --> (n*4, 512)
        embeddings = self.cls(x_batch, z_batch)

        # ok, now create a separate axis again for the 1+3 views
        # (n*4, 512) --> (n, 4, 512)
        embeddings = embeddings.view((-1, 4, embeddings.size(1)))

        # (:, 0, :) = the anchor
        # (:, 1:, :) = the three test images
        anchor = embeddings[:, 0]
        queries = embeddings[:, 1:]

        # y_batch denotes the +ve examples
        # we need to also extract the two -ve examples
        pos_idxs = y_batch
        idcs = torch.arange(n)
        pos = queries[idcs, pos_idxs]
        neg1 = queries[idcs, (pos_idxs+1) % 3]
        neg2 = queries[idcs, (pos_idxs+2) % 3]

        loss = self.loss_fn(anchor, pos, neg1) + \
               self.loss_fn(anchor, pos, neg2)

        loss.backward()
        self.optim['cls'].step()

        with torch.no_grad():
            # Compute the distances between (on a per mini-batch)
            # basis: anchor-vs-pos, anchor-vs-neg1, anchor-vs-neg2
            d_anchor_pos = torch.sum((anchor-pos)**2, dim=1, keepdim=True)
            d_anchor_neg1 = torch.sum((anchor-neg1)**2, dim=1, keepdim=True)
            d_anchor_neg2 = torch.sum((anchor-neg2)**2, dim=1, keepdim=True)

            dd = torch.cat((d_anchor_pos, d_anchor_neg1, d_anchor_neg2), dim=1)
            acc = (dd.min(dim=1)[1] == 0).float().mean()

        losses = {}
        losses['loss'] = loss.item()
        losses['acc'] = acc.item()

        outputs = {
        }

        return losses, outputs

    def eval_on_instance(self,
                         x_batch,
                         y_batch,
                         **kwargs):
        self._eval()
        with torch.no_grad():
            preds = self.predict(x_batch)
            acc = (preds == y_batch).float().mean()

            losses = {}
            losses['acc'] = acc.item()
            outputs = {}

            return losses, outputs
