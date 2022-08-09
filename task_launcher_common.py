import argparse
import os
import numpy as np

from src.models import setup_logger
logger = setup_logger.get_logger()

import torch
from torch import nn
from torch import optim
from torch.utils.data import (DataLoader,
                              Subset)
from torchvision.models import resnet101
    
import json
import inspect
from importlib import import_module

from src.models.resnet_encoder import ResnetEncoder
from src.models.holo_encoder import HoloEncoder
from src.models.holo_contrastive_encoder import HoloContrastiveEncoder
from src.common import load_dataset

#from handlers import (rot_handler,
#                      kpt_handler,
#                      angle_analysis_handler,
#                      image_handler_default)
from src.tools import (generate_name_from_args,
                       count_params,
                       line2dict)

from src.exp_utils import insert_defaults

DEFAULTS = {
    'class': 'resnet_encoder',
    'dataset': 'celeba',
    'dataset_args': None,
    'arch': None,
    'arch_args': None,

    # Keep for now
    'arch_checkpoint': None,
    
    # If true, do not use camera to rotate volume
    'disable_rot': False,
    # If true, assume we do know the canonical camera coords
    # (the question viewpoint) and use it to rotate volume.
    # This is only used for debugging, we do not use this
    # in the experiments.
    'supervised': False,
    # If we need to produce a subset of the training set.
    'subset_train': None,
    'probe': None,
    'probe_args': None,
    'img_size': 224,
    # Use imagenet mean/std scaling
    'imagenet_scaling': False,
    'batch_size': 32,
    'epochs': 200,
    'n_channels': 3,
    # Classification loss (cce = categorical x-entropy)
    'cls_loss': 'cce',


    'contrastive': {
        # ntx is only supported for now
        'ctr_loss': 'ntx',
        # If this is set to true, then we set x1=x2 during
        # contrastive training. This means we only ever
        # contrast T(x1) and T(x1) => '2d only' data aug,
        # where T() is the 2d data augmentation function.
        'remove_x2': False,
        # temperature term for contrastive loss
        'tau': 1.0,
        # old artifact, currently not used
        'lamb': 0.0,
        # old artifact, currently not used
        'n_grad_accum': 1,
        # normalise embeddings before passing into contrastive
        # loss. should be set to true.
        'normalise': False,
        'probe_only': False,
        #'extra_metrics': None,
        'rot_aug': False
    },    

    'optim': 'adam',
    'lr': 3e-4,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.0,
    'seed': 0,
    
    'save_every': 5,
    'val_batch_size': 64,
    'num_workers': 4,
    'pin_memory': False,
    'shuffle_valid': False,
    'load_nonstrict': False,
}

def validate_args(args):
    assert args['class'] in ['resnet_encoder', 'holo_encoder', 'holo_encoder_contrastive']
    if args['class'] != 'holo_encoder_contrastive':
        # If we're not using a contrastive model, just set
        # args['contrastive'] = None, no need to fill that
        # dictionary up.
        # TODO: this might be better to somehow implement in
        # insert_defaults.
        args['contrastive'] = None
    assert type(args['disable_rot']) is bool
    assert type(args['supervised']) is bool
    assert type(args['probe']) is str
    assert type(args['probe_args']) is str
    assert type(args['img_size']) is int
    assert type(args['imagenet_scaling']) is bool
    assert type(args['batch_size']) is int
    assert type(args['epochs']) is int
    assert type(args['n_channels']) is int
    assert type(args['cls_loss']) is str
    assert args['contrastive'] is None or type(args['contrastive']) is dict
    assert type(args['optim']) is str

def import_module_dynamic(path, prefix="src."):
    mod = import_module(prefix + \
                        path.replace("/", ".").\
                        replace(".py", ""))
    return mod

def get_resnet_backbone_pretrained():
    enc = resnet101(pretrained=True)
    layers = [
        enc.conv1,
        enc.bn1,
        enc.relu,
        enc.maxpool,
    ]
    for i in range(3):
        layer_name = 'layer%d' % (i + 1)
        layers.append(getattr(enc, layer_name))
    enc = torch.nn.Sequential(*layers)
    return enc

def setup(exp_dict, savedir, batch_size=None, num_workers=0, auto_resume=True, only_valid_keys=True, load_test=False):
    
    insert_defaults(exp_dict, DEFAULTS, only_valid_keys=only_valid_keys, verbose=True)
    validate_args(exp_dict)

    #if exp_dict['mode'] == 'train':
    torch.manual_seed(exp_dict['seed'])
    #else:
    #    torch.manual_seed(0)

    # imagenet scaling. Otherwise, the argument
    # --imagenet_scaling will define this.
    if exp_dict['arch'] is None:
        # Assume we're using imagenet pretrained
        logger.info("arch==None, so use imagenet scaling...")
        imagenet_scaling = True
    else:
        if not exp_dict['imagenet_scaling']:
            imagenet_scaling = False
        else:
            logger.info("imagenet_scaling==True, so use imagenet scaling...")
            imagenet_scaling = True

    if exp_dict['dataset_args'] is not None:
        dataset_args = eval(exp_dict['dataset_args'])
    else:
        dataset_args = {}

    if batch_size is not None:
        bs = batch_size
    else:
        bs = exp_dict['batch_size']
    #if exp_dict['mode'] == 'train':
    #    bs = exp_dict['batch_size']
    #else:
    #    bs = exp_dict['val_batch_size']
    
    loader_dict = {}
    if load_test:    
        ds_test = load_dataset(
            name=exp_dict['dataset'],
            img_size=exp_dict['img_size'],
            imagenet_scaling=imagenet_scaling,
            train=False,
            **dataset_args
        )
        loader_test = DataLoader(
            ds_test,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )
        loader_dict['test'] = loader_test
    else:            
        ds_train, ds_valid = load_dataset(
            name=exp_dict['dataset'],
            img_size=exp_dict['img_size'],
            imagenet_scaling=imagenet_scaling,
            train=True,
            **dataset_args
        )
        loader_train = DataLoader(
            ds_train,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=exp_dict['pin_memory']
        )
        # ** NOTE: if you are doing
        # contrastive experiments, in the 2nd stage of training
        # MAKE SURE that shuffle=True. If it's not, the valid
        # SimCLR loss will be quite inflated. This must be
        # because the mean SimCLR loss (over all minibatches)
        # is actually dependent on the contents inside the minibatch
        # (due to the implicit softmax of the batch axis) and a
        # clevr_kiwi_x dataset minibatch can have repeat images
        # due to many questions inside it mapping to the same scene.
        loader_valid = DataLoader(
            ds_valid,
            batch_size=bs,
            shuffle=True if exp_dict['shuffle_valid'] else False,
            num_workers=num_workers,
            pin_memory=exp_dict['pin_memory']
        )
        loader_dict['train'] = loader_train
        loader_dict['valid'] = loader_valid

    if exp_dict['subset_train'] is not None:
        # The subset is randomly sampled from the
        # training data, and changes depending on
        # the seed.
        indices = np.arange(0, exp_dict['subset_train'])
        rs = np.random.RandomState(exp_dict['seed'])
        rs.shuffle(indices)
        indices = indices[0:exp_dict['subset_train']]
        old_ds_train = ds_train
        ds_train = Subset(old_ds_train, indices=indices)
        # Transfer over vocab file
        ds_train.vocab = old_ds_train.vocab

    logger.info("Save path: {}".format(savedir))
    
    enc = None
    if exp_dict['arch'] is None:
        """
        https://github.com/ethanjperez/film/blob/master/scripts/extract_features.py
        Based on this: take resnet101, and take up to
        resblock #4.
        """
        logger.info("`arch` is `None`, so defaulting to Resnet-101 pretrained...")
        enc = get_resnet_backbone_pretrained()
    else:
        logger.info("Loading architecture %s..." % \
                    (exp_dict['arch']))
        mod = import_module_dynamic(exp_dict['arch'])
        arch_kwargs = eval(exp_dict['arch_args'])
        dd = mod.get_network(**arch_kwargs)
        gen = dd['gen']
        if exp_dict['arch_checkpoint'] is not None:
            chkpt_dat = torch.load(exp_dict['arch_checkpoint'])
            logger.info("  Loading checkpoint %s" % exp_dict['arch_checkpoint'])
            gen.load_state_dict(chkpt_dat['enc'],
                                strict=False if exp_dict['load_nonstrict'] else True)
        if not hasattr(gen, 'encode'):
            raise Exception("gen must have `encode` method!!")
        logger.info("Encoder:")
        logger.info(str(gen).replace("\n","\n  "))
        logger.info("# trainable params: %i" % count_params(gen))
        logger.info("# total params: %i" % count_params(gen, False))
        enc = gen

    # Put on GPU and also eval mode.
    enc.cuda()
    enc.eval()

    probe = None
    if exp_dict['probe'] is not None:
        logger.info("Importing probe: %s" % exp_dict['probe'])
        mod = import_module_dynamic(exp_dict['probe'])
        logger.info("Probe supports the following args: {}".\
            format(str(tuple(inspect.getargspec(mod.get_network).args))))
        probe_args = eval(exp_dict['probe_args']) if exp_dict['probe_args'] is not None else {}
        #if exp_dict['arch_checkpoint'] is not None:
        #    print("  `n_in` of probe must match `enc_dim` of arch " + \
        #          "so performing this replacement...")
        #    probe_exp_dict['n_in'] = arch_kwexp_dict['enc_dim']
        if load_test:
            vocab = ds_test.vocab
        else:
            vocab = ds_train.vocab
        probe = mod.get_network(vocab, **probe_args)
        logger.info("** PROBE ** ")
        logger.info("  " + str(probe).replace("\n","\n  "))
        logger.info("# trainable params: %i" % count_params(probe))
        logger.info("# total params: %i" % count_params(probe, False))
        
        # HACK: this is needed because of a stupid choice I made with
        # the probe class. It internally uses a cam_encode_3d which is
        # only used for holo_contrastive_encoder. If we're not in 
        # contrastive mode, change it to nn.Identity() so that chkpt
        # loading later on does not throw an error.
        if exp_dict['class'] != "holo_encoder_contrastive":
            logger.debug("probe.cam_encode_3d = nn.Identity()")
            probe.cam_encode_3d = nn.Identity()        
    else:
        if exp_dict['class'] == 'resnet_encoder':
            raise Exception("probe must be specified for `resnet_encoder` class!")

    if exp_dict['class'] == 'resnet_encoder':
        net = ResnetEncoder(
            enc=enc,
            probe=probe,
            cls_loss=exp_dict['cls_loss'],
            opt_args={'lr': exp_dict['lr'],
                      'betas': (exp_dict['beta1'], exp_dict['beta2']),
                      'weight_decay': exp_dict['weight_decay']},
            handlers=[]
        )

    else:

        if exp_dict['optim'] == 'sgd':
            opt_class = optim.SGD
            opt_args = {'lr': exp_dict['lr'],
                        'weight_decay': exp_dict['weight_decay']}
        elif exp_dict['optim'] == 'adam':
            opt_class = optim.Adam
            opt_args = {'lr': exp_dict['lr'],
                        'betas': (exp_dict['beta1'], exp_dict['beta2']),
                        'weight_decay': exp_dict['weight_decay'] }
        else:
            try:
                from apex.optimizers import FusedAdam
                opt_class =  FusedAdam
                opt_args = {'lr': exp_dict['lr'],
                            'betas': (exp_dict['beta1'], exp_dict['beta2']),
                            'weight_decay': exp_dict['weight_decay'] }
            except:
                raise Exception("Import `apex` failed. This is needed for fused optimiser")

        logger.info("Optimiser class: " + str(opt_class))
        logger.info("Optimiser args: " + str(opt_args))

        extra_kwargs = {}
        if exp_dict['class'] == 'holo_encoder':
            class_name = HoloEncoder
            extra_kwargs = {}
        else:
            class_name = HoloContrastiveEncoder            
            extra_kwargs = exp_dict['contrastive']
            logger.info("Contrastive mode kwargs: {}".format(extra_kwargs))

        net = class_name(
            enc=enc,
            probe=probe,
            disable_rot=exp_dict['disable_rot'],
            #disable_train_enc=exp_dict['disable_train_enc'],
            cls_loss=exp_dict['cls_loss'],
            supervised=exp_dict['supervised'],
            opt=opt_class,
            opt_args=opt_args,
            handlers=[],
            ignore_last_epoch=False,
            **extra_kwargs
        )

    if exp_dict['load_nonstrict']:
        net.load_strict = False

    if exp_dict['arch_checkpoint'] is None:
        if auto_resume:
            logger.info("arch_checkpoint=None, default to finding latest chkpt...")
            # Resume experiment if it exists
            LATEST_MODEL = "{}/model.pth".format(savedir)
            BEST_MODEL = "{}/model_best.pth".format(savedir)
            if os.path.exists(LATEST_MODEL):
                net.load(LATEST_MODEL)
                logger.info("Loaded: {}".format(LATEST_MODEL))
            elif os.path.exists(BEST_MODEL):
                net.load(BEST_MODEL)
                logger.info("Loaded: {}".format(BEST_MODEL))
            else:
                #logger.info("Could not find pre-trained chkpt, training from scratch...")
                pass
    else:
        logger.info("arch_checkpoint is defined, loading: {}".\
            format(exp_dict['arch_checkpoint']))
        net.load(exp_dict['arch_checkpoint'])

    return net, loader_dict

    """
    elif exp_dict['mode'] == 'dump_imgs':

        batch = iter(loader_train).next()[0]
        from torchvision.utils import save_image

        save_image(batch*0.5 + 0.5, "batch.png")

    elif exp_dict['mode'] == 'onnx':

        xb = iter(loader_train).next()[0]

        torch.onnx.export(probe, xb,
                          "%s/probe.onnx" % expt_dir,
                          verbose=1)

    elif exp_dict['mode'] in ['eval_valid', 'eval_test']:

        from tqdm import tqdm

        if exp_dict['mode'] == 'eval_valid':
            print("Evaluating on valid set...")
            loader = loader_valid
            ds = ds_valid
        else:
            print("Evaluating on test set...")
            ds = load_dataset(
                name=exp_dict['dataset'],
                img_size=exp_dict['img_size'],
                imagenet_scaling=imagenet_scaling,
                train=False
            )
            loader = DataLoader(ds,
                                batch_size=exp_dict['val_batch_size'],
                                shuffle=False,
                                num_workers=exp_dict['num_workers'])

        net_epoch = net.last_epoch

        # Ok, for each possible camera view, create that valid set
        # and evaluate on it.
        preds = []
        gt = []
        tfs = []
        cams = []
        nc = []
        ns = []
        nm = []
        n_obj = []

        pbar = tqdm(total=len(loader))
        for batch in loader:
            batch = net.prepare_batch(batch)
            pred = net.predict(*batch).argmax(dim=1)
            y_batch = batch[-3]
            meta_batch = batch[-1]
            cam_xyz_batch = batch[3][:,0:3]

            preds.append(pred)
            gt.append(y_batch)
            cams.append(cam_xyz_batch)
            #dists.append(meta_batch['d_from_cc'])
            tfs += meta_batch['template_filename']
            nc.append(meta_batch['n_color_unique'])
            ns.append(meta_batch['n_shape_unique'])
            nm.append(meta_batch['n_mat_unique'])
            n_obj.append(meta_batch['n_objects'])

            pbar.update(1)
        pbar.close()

        acc_ind = (torch.cat(preds, dim=0) == torch.cat(gt, dim=0)).float().cpu().numpy()
        cams = torch.cat(cams, dim=0).cpu().numpy()
        nc = torch.cat(nc, dim=0).cpu().numpy()
        ns = torch.cat(ns, dim=0).cpu().numpy()
        nm = torch.cat(nm, dim=0).cpu().numpy()
        n_obj = torch.cat(n_obj, dim=0).cpu().numpy()

        with open("%s/%s.%i.csv" % (expt_dir, exp_dict['mode'], net_epoch), "w") as f:
            f.write("correct,cam_x,cam_y,cam_z,tf,n_color_unique,n_shape_unique,n_mat_unique,n_obj\n")
            for j in range(len(cams)):
                f.write("%f,%f,%f,%f,%s,%i,%i,%i,%i\n" % \
                        (acc_ind[j], cams[j][0], cams[j][1], cams[j][2], tfs[j], nc[j], ns[j], nm[j], n_obj[j]))
    """