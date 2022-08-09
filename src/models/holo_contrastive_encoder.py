import torch
import numpy as np
from collections import OrderedDict
from torch import optim
from torch.nn import functional as F
from itertools import chain
from .base import Base
from .holo_ae import HoloAE
from torch import nn
from torch.distributions import (Uniform,
                                 Normal,
                                 Dirichlet)

from .holo_encoder import HoloEncoder
from .util.nxent import contrastive_loss
from .util import rotation as util_rot
from .util.hobbit_clr import SupConLoss

from .setup_logger import get_logger
logger = get_logger()

class HoloContrastiveEncoder(HoloEncoder):
    """
    """

    def __init__(self, *args, **kwargs):
        self.ctr_loss = kwargs.pop('ctr_loss')
        self.probe_only = kwargs.pop('probe_only')
        if self.probe_only:
            logger.warning("self.probe_only==True, this should only be set if " + \
                           "you want to perform second stage (FILM) training.")

        self.norm = kwargs.pop('normalise')
        self.tau = kwargs.pop('tau')
        self.n_grad_accum = kwargs.pop('n_grad_accum')

        # Use if you do not want to use multi-view (3D)
        # during training.
        self.remove_x2 = kwargs.pop('remove_x2')
        self.rot_aug = kwargs.pop('rot_aug')
        self.lamb = kwargs.pop('lamb')

        if self.ctr_loss == 'ntx':
            #self.embed_loss = contrastive_loss(tau=self.tau)
            self.embed_loss = SupConLoss(temperature=self.tau,
                                         contrast_mode='one')
        else:
            raise Exception("no other loss supported currently")

        # We may want to log extra metrics which do not contribute to
        # the training loss. These will make iterations slower but
        # would be otherwise useful for analysis≥
        #extra_metrics = kwargs.pop('extra_metrics', 'none')        
        
        #if extra_metrics not in ['rot_all', 'rot_y', 'none']:
        #    raise ValueError("`extra_metrics` must be either `rot_all`, `rot_y`, or `none`")
        #logger.info("self.extra_metrics =" + str(extra_metrics))
        #self.extra_metrics = extra_metrics

        super(HoloContrastiveEncoder, self).__init__(*args, **kwargs)

        if self.supervised:
            raise Exception("`supervised` option is not used for this class")

        self.NUM_TESTING_ITERS = 20
        if self.NUM_TESTING_ITERS > 0:
            logger.warning("NUM_TESTING_ITERS > 0, make sure that `shuffle_valid` is also set")

        if self.enc.postprocessor is not None:
            logger.info("Found `postprocessor` module in encoder:")
            logger.info(str(self.enc.postprocessor))
            num_params = sum([ np.product(p.shape) for p in \
                               self.enc.postprocessor.parameters()])
            logger.info("Number of parameters in postprocessor: %i" \
                        % num_params)

    """
    def _extract_angles_and_offsets(self, x):

        ns = x[:, 0:3]
        ts = x[:, 3:6]
        theta_mu = x[:, 6:7]
        theta_std = x[:, 7:]
        #if self.beta > 0:
        #    theta = Normal(theta_mu, torch.exp(theta_std))
        #else:
        theta = Normal(theta_mu, theta_std*0.)

        nz = ns[:, 0:1]
        ny = ns[:, 1:2]
        nx = ns[:, 2:3]

        tx = ts[:, 0:1]
        ty = ts[:, 1:2]
        tz = ts[:, 2:3]

        dd = {
            'nz': nz, 'ny': ny, 'nx': nx,
            'tx': tx, 'ty': ty, 'tz': tz,
            'theta': theta
        }
        return dd
    """

    # TODO: this is also defined in the base class,
    # but using weird zero padding on the edges.
    def stn(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        if x.is_cuda:
            grid = grid.to(self.device)
        out = F.grid_sample(x, grid, padding_mode='zeros')
        return out

    def _accuracy_from_encodings(self, enc1, enc2):
        with torch.no_grad():
            # Compute accuracy here. Out of n**2 predictions,
            # find out how many of the top n predictions are
            # on the diagonal.
            bs = enc1.size(0)
            logits = torch.exp(torch.matmul(enc1, enc2.transpose(0, 1)) / self.tau)
            logits = logits.flatten()
            diag_indices = np.where(np.eye(bs).flatten() == 1)[0]
            topk = torch.topk(logits, k=bs).indices.cpu().numpy()
            return len(set(diag_indices).intersection(set(topk))) / bs

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

        losses = {}

        if self.probe_only:

            if train:
                for key in self.optim:
                    self.optim[key].zero_grad()

            self.enc.eval()
            self.enc.postprocessor.train()

            # `probe_only` refers to the second stage of training.
            # In the first stage, we trained the contrastive
            # encoder. In this stage, we use the representation
            # (+ with a learned postprocessor, optionally) and
            # use it to train on FILM. The encoder's parameters
            # still stay fixed, apart from the optional postprocessor.
            # The postprocessor can be added in the second stage
            # by changing the architecture but adding `--load_nonstrict`
            # in the task launcher.

            # Do most of the work under no_grad to save
            # on memory.
            with torch.no_grad():
                h1, _ = self.enc.encode(x_batch)
                if kwargs['iter'] < self.NUM_TESTING_ITERS:
                    # (To verify, also measure the triplet loss here
                    # but only do it for the first epoch since it's
                    # expensive to compute, and we're only interested
                    # in verifying the model's loss is still the same
                    # as in the first stage of training.)
                    h2, _ = self.enc.encode(x2_batch)
                    enc1 = self.enc.vol2enc(h1)
                    enc2 = self.enc.vol2enc(h2)
                    if self.norm:
                        enc1 = F.normalize(enc1, dim=1)
                        enc2 = F.normalize(enc2, dim=1)
                    enc_total = torch.stack((enc1, enc2), dim=1)
                    triplet_loss = self.embed_loss(enc_total)
                    losses['acc'] = self._accuracy_from_encodings(enc1, enc2)
                    losses['triplet_loss'] = triplet_loss.item()

            # Ok, now run h1 through the postprocessor
            # module. We only detach before it, so if
            # a postprocessor does exist, then it will
            # be updated through backprop.
            h1, _ = self.enc.encode(x_batch)
            h1 = self.enc.postprocess(h1.detach())
            
            if kwargs['iter'] == 1:
                logger.info("h1 shape = {}".format(h1.shape))
                #logger.info("postprocessor = {}".format(self.enc.postprocess))

        else:

            if train:
                if 'probe' in self.optim:
                    self.optim['probe'].zero_grad()
                # We don't zero the g grad here
                # since we may be doing grad
                # accumulation.

            if self.remove_x2 and train:
                # Set this option if you want to see how the contrastive
                # model trains when training does not have concurrent
                # access to two different views of the same scene.
                # In validation however, we will still be using
                # different views of the same scene.
                if kwargs['iter'] == 1:
                    logger.info("`remove_x2`==`True` and we are in training mode, " + \
                                "so replace x2_batch with x1_batch...")
                x2_batch = x_batch

            # Contrastive loss between examples that have
            # undergone random 2D data augmentation.
            h1, _y12 = self.enc.encode(x_batch)
            h2, _y21 = self.enc.encode(x2_batch)
            enc1 = self.enc.vol2enc(h1)
            enc2 = self.enc.vol2enc(h2)
            if self.norm:
                enc1 = F.normalize(enc1, dim=1)
                enc2 = F.normalize(enc2, dim=1)

            if self.rot_aug:
                # Minimise distance between h1 and its
                # random rotation on the y axis.
                bs = x_batch.size(0)
                rnd_y = torch.ones((bs, 1)).uniform_(-np.pi, np.pi)
                rnd = torch.cat((rnd_y*0, rnd_y, rnd_y*0), dim=1).float().cuda()
                R_mat = util_rot.t_get_theta(rnd)
                h1_rot = self.stn(h1, R_mat)
                enc1_rot = self.enc.vol2enc(h1_rot)
                if self.norm:
                    # enc1 is already normed
                    enc1_rot = F.normalize(enc1_rot, dim=1)
                triplet_rot_aug_loss = self.embed_loss(
                    torch.stack((enc1, enc1_rot), dim=1)
                )
                losses['triplet_rot_aug_loss'] = triplet_rot_aug_loss.item()

            """
            if self.extra_metrics != 'none':
                if torch.no_grad():
                    bs = x_batch.size(0)
                    if self.extra_metrics == 'rot_y':
                        # Only sample rotations on yaw.
                        rnd_y = torch.ones((bs, 1)).uniform_(-np.pi, np.pi)
                        rnd = torch.cat((rnd_y*0, rnd_y, rnd_y*0), dim=1).float().cuda()
                    else:
                        # Sample on all axes.
                        rnd = torch.ones((bs, 3)).uniform_(-np.pi, np.pi).float().cuda()
                    R_mat = util_rot.t_get_theta(rnd)
                    h1_rot = self.stn(h1, R_mat)
                    enc1_rot = self.enc.vol2enc(h1_rot)
                    if self.norm:
                        # enc2 is already normed
                        enc1_rot = F.normalize(enc1_rot, dim=1)
                    enc_rot_total = torch.stack((enc1_rot, enc2), dim=1)
                    triplet_rot_loss = self.embed_loss(enc_rot_total)
                    losses['triplet_%s_loss' % self.extra_metrics] = \
                        triplet_rot_loss.item()
                    losses['%s_acc' % self.extra_metrics2] = \
                        self._accuracy_from_encodings(enc1_rot, enc2)
            """

            if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
                logger.info("h1 shape: " + str(h1.shape))
                logger.info("h2 shape: " + str(h2.shape))
                logger.info("enc1 shape: " + str(enc1.shape))
                logger.info("enc2 shape: " + str(enc2.shape))
                logger.info("effective batch size: %i" % (self.n_grad_accum*h1.size(0)))

            # <anchor, pos, neg>
            if self.ctr_loss == 'ntx':
                #h1 = h1.view(h1.size(0), -1)
                #h2 = h2.view(h2.size(0), -1)
                #triplet_loss = self.embed_loss(enc1, enc2)
                enc_total = torch.stack((enc1, enc2), dim=1)
                triplet_loss = self.embed_loss(enc_total)
                if self.rot_aug:
                    triplet_loss = triplet_loss + triplet_rot_aug_loss
                losses['acc'] = self._accuracy_from_encodings(enc1, enc2)
                if self.rot_aug:
                    losses['acc_rot_aug'] = self._accuracy_from_encodings(enc1, enc1_rot)
            else:
                raise Exception("Unknown `ctr_loss`!`")

            gen_loss = (triplet_loss) / self.n_grad_accum #+ self.rot_consist*consist_loss
            losses['triplet_loss'] = triplet_loss.item()

            if _y12 is not None and self.lamb > 0:
                if self.supervised:
                    #gen_loss = gen_loss + self.lamb*sup_recon_loss
                    #losses['sup_recon_loss'] = sup_recon_loss.item()
                    raise NotImplementedError()
                else:
                    #gen_loss = gen_loss + self.lamb*unsup_recon_loss
                    #losses['unsup_recon_loss'] = unsup_recon_loss.item()
                    gen_loss = gen_loss + self.lamb*triplet_rot_loss
                    losses['triplet_rot_loss'] = triplet_rot_loss.item()

            if train:
                gen_loss.backward()
                if kwargs['iter'] % self.n_grad_accum == 0:
                    self.optim['g'].step()
                    self.optim['g'].zero_grad()

        if self.probe is not None:

            if not self.disable_rot:
                if kwargs['iter'] == 1:
                    logger.info("`self.disable_rot == False` so encoding camera...")
                theta = self.probe.cam_encode_3d(cam_batch)
                rot_mat = util_rot.t_get_theta(theta[:, 0:3], theta[:, 3:6])
                if kwargs['iter'] == 1 and kwargs['epoch'] == 1:
                    logger.info("`rot_mat` shape: " + str(rot_mat.shape))

                h_probe = self.stn(h1, rot_mat) # actual viewpoint
                with torch.no_grad():
                    losses['probe_pred_theta_z'] = theta[:, 0].mean().item()
                    losses['probe_pred_theta_y'] = theta[:, 1].mean().item()
                    losses['probe_pred_theta_x'] = theta[:, 2].mean().item()
                    losses['probe_pred_t_x'] = theta[:, 3].mean().item()
                    losses['probe_pred_t_y'] = theta[:, 4].mean().item()
                    losses['probe_pred_t_z'] = theta[:, 5].mean().item()
            else:
                h_probe = h1

            probe_out = self.probe(h_probe,
                                   q_batch,
                                   cam_batch)
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
            h, _ = self.enc.encode(x_batch)
            if not self.disable_rot:
                theta = self.probe.cam_encode_3d(cam_batch)
                rot_mat = util_rot.t_get_theta(theta[:, 0:3], theta[:, 3:6])
                h_probe = self.stn(h, rot_mat) # actual viewpoint
            else:
                h_probe = h
            probe_out = self.probe(h_probe, q_batch, cam_batch)
            return probe_out

    def save(self, filename, epoch):
        dd = {}
        # Save the models.
        dd['enc'] = self.enc.state_dict()
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
        self.enc.load_state_dict(dd['enc'], strict=self.load_strict)
        if self.probe is not None:
            if 'probe' in dd:
                self.probe.load_state_dict(dd['probe'],
                                           strict=self.load_strict)
        # Load the models' optim state.
        for key in self.optim:
            if ('optim_%s' % key) in dd:
                self.optim[key].load_state_dict(dd['optim_%s' % key])
        if not self.ignore_last_epoch:
            self.last_epoch = dd['epoch']
        else:
            logger.warning("Last epoch indicated in chkpt is {} ".format(dd['epoch']) + \
                            "but since `ignore_last_epoch`==`True` we ignore it here...")
