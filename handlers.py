import torch
import pickle
import os
import numpy as np
#import pretrained
from torchvision.utils import save_image
from sklearn.linear_model import LinearRegression

def image_handler_default(dest_dir,
                          save_images_every):
    def _image_handler_default(losses, batch, outputs, kwargs):
        if outputs == {}:
            return {}
        if kwargs['iter'] == 1 and kwargs['mode'] == 'train':
            if kwargs['epoch'] % save_images_every == 0:
                mode = kwargs['mode']

                # TODO: do for valid as well
                epoch = kwargs['epoch']

                input_ = outputs['input']*0.5 + 0.5
                input2 = outputs['input2']*0.5 + 0.5
                recon = outputs['recon']*0.5 + 0.5
                x_undo = outputs['x_undo']*0.5 + 0.5
                x_h1_onto_h2 = outputs['x_h1_onto_h2']*0.5 + 0.5

                # input image
                # reconstruction
                # input 2 (another view of input image)
                # (4) encode input image, undo rotation
                # (5) transform (4) into input 2
                imgs = torch.cat([input_, recon, input2, x_undo, x_h1_onto_h2], dim=0)
                save_image( imgs,
                            nrow=input_.size(0),
                            filename="%s/%i_%s.png" % (dest_dir, epoch, mode))
        return {}
    return _image_handler_default

def image_handler_2d(dest_dir,
                     save_images_every):
    def _image_handler_2d(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1:
            if kwargs['epoch'] % save_images_every == 0:
                mode = kwargs['mode']
                epoch = kwargs['epoch']
                inputs = outputs['input']*0.5 + 0.5
                recon = outputs['recon']*0.5 + 0.5
                imgs = torch.cat([inputs, recon], dim=0)
                save_image( imgs,
                            nrow=inputs.size(0),
                            filename="%s/%i_%s.png" % (dest_dir, epoch, mode))
        return {}
    return _image_handler_2d


def image_handler_gan(dest_dir,
                      save_images_every):
    def _image_handler_gan(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and kwargs['mode'] == 'train':
            if kwargs['epoch'] % save_images_every == 0:
                mode = kwargs['mode']

                # TODO: do for valid as well
                epoch = kwargs['epoch']

                xreal = outputs['x_real']*0.5 + 0.5
                xrecon = outputs['x_recon']*0.5 + 0.5

                xfake = outputs['x_fake']*0.5 + 0.5
                rots = [ x*0.5 + 0.5 for x in outputs['rot'] ]
                save_image( torch.cat([xreal, xrecon] + [xfake] + rots, dim=0),
                            nrow=xfake.size(0),
                            filename="%s/%i_%s.png" % (dest_dir, epoch, mode))
        return {}
    return _image_handler_gan


'''
def rot_handler(gan,
                cls,
                axis,
                loader,
                dest_dir):
    def _rot_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and kwargs['mode'] == 'train':
            # NOTE: this only works on the speedy dataset.

            # Get the first element in the non-shuffled
            # batch, which will correspond to the chair
            # at rotation angle 0. Then rotate this
            # all the way to 360, at each way measuring
            # the predicted angle output by the classifier.

            gan._eval()

            #dest_dir = "%s/%s/rot_preds" % (save_path, name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            bs = loader.batch_size
            n_batches = int(np.ceil(len(loader.dataset.angles) / bs))

            cls_pretrained = cls

            # Get first element.
            x_batch, _ = iter(loader).next()
            x_batch = x_batch[0:1].repeat(bs, 1, 1, 1)

            if gan.use_cuda:
                x_batch = x_batch.cuda()

            output = None
            gan_angles = np.linspace(gan.angles['min_angle_%s' % axis],
                                     gan.angles['max_angle_%s' % axis],
                                     num=len(loader.dataset.angles))
            with torch.no_grad():
                # Encode it, do the rotations
                enc, _, h = gan.generator.encode(x_batch)
                # rotate h here
                #h = gan.rotate(h, angles_)

                # HACK: we assume it's the pitch axis
                preds = []
                for j in range(n_batches):
                    this_angles = gan_angles[j*bs:(j+1)*bs]
                    if len(this_angles) < bs:
                        h = h[0:len(this_angles)]
                        enc = enc[0:len(this_angles)]
                    angles = np.zeros((h.size(0), 3)).astype(np.float32)
                    angles[:, gan.rot2idx[axis] ] += this_angles
                    angles = torch.from_numpy(angles).float().cuda()
                    h_rot = gan.rotate(h, angles)
                    dec_enc_rot = gan.generator.decode(enc, h_rot)
                    preds.append( cls_pretrained(dec_enc_rot) )
                preds = torch.cat(preds, dim=0).cpu().numpy().flatten()
                preds = preds.astype(np.float32)
                gt = np.asarray(loader.dataset.angles)
                error = (preds-gt)**2
                with open("%s/%i.pkl" % (dest_dir, kwargs['epoch']), "wb") as f:
                    pickle.dump(preds, f)
                output = {'pred_rot_mean': error.mean().item(),
                          'pred_rot_std': error.std().item()}
            return output
        return {}
    return _rot_handler
'''

def rot_handler(gan,
                cls_siamese,
                cls_pose,
                axis,
                min_angle_gt,
                max_angle_gt,
                num_interps,
                loader,
                dest_dir,
                img_format='png'):
    '''This handler works as follows:

    # TODO: rewrite this documentation

    - The specific data loader here only returns frontal
    images, as determined by the kpts2pose model.
    - Given a batch of frontal images, use the siamese
    classifier to measure p(same|x1,x_i), where x_i (i=1..N)
    is continually rotated from its initial pos (0 deg)
    to -60. Also do the same for 0 to +60. Sum up these
    p's (the best value you can attain is theoretically N).
    - We also use a pre-trained network to predict pose
    from image. We want to measure the MAE between the
    angles we use (e.g. [0,...,-60] and [0,...60]) and
    the prediction from the network.
    '''
    def _rot_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and kwargs['mode'] == 'train':
            # NOTE: this only works on the speedy dataset.

            gan._eval()

            #dest_dir = "%s/%s/rot_preds" % (save_path, name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # Let's sample `bs` number of images
            # whose poses are frontal (0 degrees)
            x_batch = gan.prepare_batch(iter(loader).next())[0]
            #x_batch, _, _ = iter(loader).next()
            #if gan.use_cuda:
            #    x_batch = x_batch.cuda()

            # The min and max of the sampling distribution.
            gan_angles = np.linspace(gan.min_angle,
                                     gan.max_angle,
                                     num=num_interps)

            losses = {}
            with torch.no_grad():

                enc, q = gan.generator.encode(x_batch, x_batch)
                h = gan.generator.enc2vol(enc)
                rz, ry, rx, tx, ty, tz = gan._extract_angles_and_offsets(q)

                if cls_siamese is not None:
                    preds_siamese = []
                if cls_pose is not None:
                    preds_pose = []
                preds_imgs = []

                # For each angle...
                for (j, theta) in enumerate(gan_angles):

                    h_rot = gan.rotate(h,
                                       rz[1], ry[1]*0 + theta, rx[1],
                                       tx[1], ty[1], tz[1])
                    dec_enc_rot = gan.generator.decode(h_rot)

                    # Measure the p(same) between rot_j(xb) and xb.
                    if cls_siamese is not None:
                        preds_siamese.append( cls_siamese(dec_enc_rot, x_batch) )
                    if cls_pose is not None:
                        preds_pose.append( cls_pose(dec_enc_rot).view(-1, 1) )
                    preds_imgs.append(dec_enc_rot)

                #  `preds_siamese` is a matrix of the form:
                #
                #     ____________________________________________________
                #     | P(x1,rot(x1, theta_1)) ... P(x1,rot(x1,theta_N))  |
                #     | .                       .                       . |
                # bs  | .                       .                       . |
                #     | .                       .                       . |
                #     | P(xn,rot(xn, theta_1)) ... P(xn,rot(xn,theta_N)). |
                #     ____________________________________________________
                #                               N
                #
                # where P(xi,xj) is the probability that (xi,xj) is from
                # the same id, according to the siamese classifier. We
                # compute the row-wise sum of this matrix, and then compute
                # the mean over this. Since the max value p() can take on
                # is 1, the best score you can get here is when the resulting
                # sum is [N, ..., N] (len of this array is `bs`), and then
                # the resulting mean is just bs*N / bs = N.

                if cls_siamese is not None:
                    preds_siamese = torch.cat(preds_siamese, dim=1).cpu().numpy().\
                        astype(np.float32)
                    losses['cls_siamese_sum'] = \
                        preds_siamese.sum(axis=1).mean()

                #  `preds_pose` is a matrix of the form:`
                #
                #     ____________________________________________
                #     | R(rot(x1,theta_1)) ... R(rot(x1,theta_N)) |
                #     | .                   .                   . |
                # bs  | .                   .                   . |
                #     | .                   .                   . |
                #     | R(rot(xn,theta_1)) ... R(rot(xn,theta_N)) |
                #     _____________________________________________
                #                          N
                #
                # where R(.) is the predicted pose from the pretrained
                # pose classifier. We compute the MAE between this
                # and `gan_angles_` (an N-vector) in row-wise fashion,
                # then compute the mean of this entire matrix.

                if cls_pose is not None:

                    # The min and max of the GT sampling distribution.
                    gt_angles = np.linspace(gan._to_radians(min_angle_gt),
                                            gan._to_radians(max_angle_gt),
                                            num=num_interps)
                    preds_pose = torch.cat(preds_pose, dim=1).cpu().numpy().\
                        astype(np.float32)
                    gt_angles_ = gt_angles.reshape(1, len(gt_angles)).\
                        repeat(len(preds_pose), axis=0)
                    gt_angles_reverse_ = gt_angles[::-1].reshape(1, len(gt_angles)).\
                        repeat(len(preds_pose), axis=0)

                    errors = (preds_pose - gt_angles_)**2
                    errors2 = (preds_pose - gt_angles_reverse_)**2

                    if errors.mean() < errors2.mean():
                        losses['cls_rot2f_mean'] = errors.mean()
                        losses['cls_rot2f_std'] = errors.std()
                    else:
                        losses['cls_rot2f_mean'] = errors2.mean()
                        losses['cls_rot2f_std'] = errors2.std()

                    #losses['cls_rot_%i_%i_mean' % (min_angle, max_angle)] = \
                    #    np.mean(np.abs(gt_angles_-preds_pose))

                # Save `preds_imgs` to disk.

                tot_imgs = torch.stack(preds_imgs, dim=0) # (100, 64, 3, 32, 32)

                nc, h, w = tot_imgs.size(2), tot_imgs.size(3), tot_imgs.size(4)
                save_image(tot_imgs.view(-1, nc, h, w)*0.5 + 0.5,
                           nrow=len(preds_imgs[0]),
                           filename="%s/%i.%s" % (dest_dir, kwargs['epoch'], img_format))

                if cls_pose is not None:
                    with open("%s/pred_pose_%i.pkl" % (dest_dir, kwargs['epoch']), "wb") as f:
                        pickle.dump(preds_pose, f)
                    with open("%s/min_max_angles_%i.txt" % (dest_dir, kwargs['epoch']), "w") as f:
                        f.write(str(gan.min_angle) + "," + str(gan.max_angle))

            return losses
        return {}
    return _rot_handler

def angle_analysis_handler(gan, loader):
    def _angle_analysis_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and kwargs['mode'] == 'train':

            gan._eval()

            pred_angles = []
            with torch.no_grad():
                for x_batch, _, _ in loader:
                    if gan.use_cuda:
                        x_batch = x_batch.cuda()
                    _, angles, _ = gan.generator.encode(x_batch)
                    pred_angles.append(angles)
            pred_angles = torch.cat(pred_angles, dim=0).cpu()
            print("pred angles shape:", pred_angles.shape)
            min_angle, max_angle = torch.min(pred_angles), torch.max(pred_angles)
            mean_angle, std_angle = torch.mean(pred_angles), torch.std(pred_angles)

            return {'angle_tot_min': min_angle.item(),
                    'angle_tot_max': max_angle.item(),
                    'angle_tot_mean': mean_angle.item(),
                    'angle_tot_std': std_angle.item()}
        return {}
    return _angle_analysis_handler

def kpt_handler(gan, loader, dest_dir):
    def _kpt_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1 and kwargs['mode'] == 'train':
            gan._eval()
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            #with torch.no_grad():
            x_batch, _, y_batch = iter(loader).next()
            im_sz = x_batch.size(-1)-1
            if gan.use_cuda:
                x_batch = x_batch.cuda()
            enc_mu, enc_std, q = gan.generator.encode(x_batch)
            enc = gan.sample_z(enc_mu, torch.exp(enc_std))[1]
            h = gan.generator.enc2vol(enc)
            pred_kpts = gan.probe(h)
            pred_kpts = (pred_kpts.reshape(-1, 68, 2).cpu()*im_sz).long()
            y_kpts = (y_batch.reshape(-1, 68, 2).cpu()*im_sz).long()

            canvas_pred = torch.zeros_like(x_batch)
            for b in range(len(pred_kpts)):
                this_pred_yx = pred_kpts[b]
                for j in range(68):
                    canvas_pred[b, :, this_pred_yx[j][1], this_pred_yx[j][0]] = 1.
            canvas_gt = torch.zeros_like(x_batch)
            for b in range(len(y_kpts)):
                this_pred_yx = y_kpts[b]
                for j in range(68):
                    canvas_gt[b, :, this_pred_yx[j][1], this_pred_yx[j][0]] = 1.

            save_image(torch.cat((canvas_gt, canvas_pred), dim=0),
                       nrow=len(canvas_gt),
                       filename="%s/kpts_%i.png" % (dest_dir, kwargs['epoch']))



        return {}
    return _kpt_handler
