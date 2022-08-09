
'''
MUST BE RUN FROM THE ROOT DIRECTORY USING:
python -m pretrained.task_launcher_celeba2pose.py ... <options>
'''

import os
import torch
import argparse
from torch import nn
from torch.utils.data import (TensorDataset,
                              DataLoader)
from torchvision import transforms
from iterators.datasets import CelebASiameseDataset
import numpy as np

from models.base import Base
from models.siamese_classifiers import (SiameseExplicitDistanceClassifier,
                                        SiameseBalancedSamplingClassifier,
                                        SiameseTripletMarginClassifier)

import pickle
import glob
import yaml
from tools import (generate_name_from_args,
                   find_latest_pkl_in_folder)
from tqdm import tqdm

from importlib import import_module

import iterators.datasets as it_ds

use_shuriken = False
try:
    from shuriken.utils import get_hparams
    use_shuriken = True
except:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--name', type=str,
                        default=None)
    parser.add_argument('--arch', type=str,
                        default=None)
    parser.add_argument('--trial_id', type=str,
                        default=None)
    parser.add_argument('--nf', type=int, default=32)
    parser.add_argument('--sigma', type=float,
                        default=0.)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float,
                        default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5,
                        help="beta1 term of ADAM")
    parser.add_argument('--beta2', type=float, default=0.99,
                        help="beta2 term of ADAM")
    parser.add_argument('--triplet', action='store_true',
                        help="Use triplet margin loss instead")
    parser.add_argument('--resume', type=str, default='auto')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--dump_preds', action='store_true')
    args = parser.parse_args()
    return args

id_ = lambda x: str(x)
KWARGS_FOR_NAME = {
    'arch': ('arch', lambda x: os.path.basename(x)),
    'img_size': ('sz', id_),
    'batch_size': ('bs', id_),
    'lr': ('lr', id_),
    'beta1': ('b1', id_),
    'beta2': ('b2', id_),
    'sigma': ('sigma', id_),
    'nf': ('nf', id_),
    'triplet': ('triplet', id_),
    'trial_id': ('_trial', id_)
}

if __name__ == '__main__':

    args = parse_args()

    print("Args from argparse:")
    print(args)

    args = vars(args)

    ######### TODO: refactor this block

    if use_shuriken:
        shk_args = get_hparams()
        print("shk args:", shk_args)
        # Stupid bug that I have to fix: if an arg is ''
        # then assume it's a boolean.
        for key in shk_args:
            if shk_args[key] == '':
                shk_args[key] = True
        args.update(shk_args)

    if args['trial_id'] is None and 'SHK_TRIAL_ID' in os.environ:
        print("SHK_TRIAL_ID found so injecting this into `trial_id`...")
        args['trial_id'] = os.environ['SHK_TRIAL_ID']

    if args['name'] is None and 'SHK_EXPERIMENT_ID' in os.environ:
        print("SHK_EXPERIMENT_ID found so injecting this into `name`...")
        args['name'] = os.environ['SHK_EXPERIMENT_ID']

    print("** ARGUMENTS **")
    print("  " + yaml.dump(args).replace("\n", "\n  "))

    print("** AUTO-GENERATED NAME **")
    name = generate_name_from_args(args, KWARGS_FOR_NAME)
    print("  " + name)

    if args['save_path'] is None:
        args['save_path'] = os.environ['RESULTS_DIR']

    # <save_path>/<seed>/<name>/_trial=<trial>,...,...,
    if args['name'] is None:
        save_path = "%s/s%i" % (args['save_path'], args['seed'])
    else:
        save_path = "%s/s%i/%s" % (args['save_path'], args['seed'], args['name'])

    print("*** SAVE PATH OF THIS EXPERIMENT: ***")
    print(save_path)
    print("*************************************")

    expt_dir = "%s/%s" % (save_path, name)
    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)

    #############################

    mod = import_module(args['arch'].replace("/", ".").\
                        replace(".py", ""))
    net = mod.get_network(n_channels=3,
                          ndf=args['nf'])

    print(net)

    train_transforms = [
        transforms.Resize(48),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    root = "%s/img_align_celeba" % os.environ['DATASET_CELEBA']
    ids_file = "%s/identity_CelebA.txt" % os.environ['DATASET_CELEBA']
    ds_train = CelebASiameseDataset(root=root,
                                    ids_file=ids_file,
                                    transforms_=train_transforms,
                                    mode='train')
    ds_valid = CelebASiameseDataset(root=root,
                                    ids_file=ids_file,
                                    transforms_=train_transforms,
                                    mode='valid')

    loader_train = DataLoader(ds_train, shuffle=True, batch_size=32)
    loader_valid = DataLoader(ds_valid, shuffle=False, batch_size=32)

    cls_kwargs = {
        'sigma': args['sigma'],
        'opt_args':{'lr': args['lr'],
                    'betas': (args['beta1'], args['beta2'])},
        'handlers': []
    }

    if args['triplet']:
        model = SiameseTripletMarginClassifier(cls=net,
                                               **cls_kwargs)
    else:
        model = SiameseBalancedSamplingClassifier(cls=net,
                                                  **cls_kwargs)
    if not isinstance(ds_train, it_ds.XXDataset):
        raise Exception("`siamese_c` requires a dataset which returns two x's")

    if args['resume'] is not None:
        if args['resume'] == 'auto':
            # autoresume
            model_dir = "%s" % expt_dir
            latest_model = find_latest_pkl_in_folder(model_dir)
            if latest_model is not None:
                print("Auto-resume mode found latest model: %s" %
                      latest_model)
                model.load(latest_model)
        else:
            print("Loading model: %s" % args['resume'])
            model.load(args['resume'])

    model.train(itr_train=loader_train,
                itr_valid=loader_valid,
                epochs=args['epochs'],
                model_dir=expt_dir,
                result_dir=expt_dir,
                save_every=args['save_every'])
