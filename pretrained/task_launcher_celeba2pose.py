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
from iterators.datasets import CelebADataset
import numpy as np
from architectures.discriminators import Discriminator_
from models.regressor import Regressor
from models.base import Base
import pickle
import glob
import yaml
from tools import (generate_name_from_args,
                   find_latest_pkl_in_folder)
from tqdm import tqdm

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
    parser.add_argument('--trial_id', type=str,
                        default=None)
    parser.add_argument('--nf', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--pose_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=512)
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
    'pose_file': ('sigma', lambda x: os.path.basename(x)),
    'nf': ('nf', id_),
    'trial_id': ('_trial', id_)
}

if __name__ == '__main__':

    args = parse_args()

    print("Args from argparse:")
    print(args)

    args = vars(args)

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

    # Just use a discriminator architecture for this.
    net = Discriminator_(
        nf=args['nf'],
        input_nc=3,
        n_out=3,
        sigmoid=False,
        spec_norm=False
    )

    print(net)

    train_transforms = [
        #transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop( (args['img_size'], args['img_size']),
                                      scale=(0.8, 1.0) ),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    print(args['pose_file'])

    if not args['dump_preds']:
        pose_arr = pickle.load(open(args['pose_file'], "rb"))
    else:
        pose_arr = None

    print("pose_arr:")
    print(pose_arr)
    print("---")

    ds_train = CelebADataset(root=os.environ['DATASET_CELEBA'] + "/img_align_celeba",
                             labels=pose_arr,
                             transforms_=train_transforms,
                             mode='train')
    ds_valid = CelebADataset(root=os.environ['DATASET_CELEBA'] + "/img_align_celeba",
                             labels=pose_arr,
                             transforms_=train_transforms,
                             mode='valid')

    loader_train = DataLoader(ds_train, shuffle=True, batch_size=args['batch_size'])
    loader_valid = DataLoader(ds_valid, shuffle=False, batch_size=args['batch_size'])

    model = Regressor(cls=net, sigma=0.)

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

    if not args['dump_preds']:

        model.train(itr_train=loader_train,
                    itr_valid=loader_valid,
                    epochs=args['epochs'],
                    model_dir=expt_dir,
                    result_dir=expt_dir,
                    save_every=args['save_every'])
    else:

        ds_all = CelebADataset(root=os.environ['DATASET_CELEBA'] + "/img_align_celeba",
                               labels=pose_arr,
                               transforms_=train_transforms,
                               mode=None)
        loader_all = DataLoader(ds_all, shuffle=False, batch_size=args['val_batch_size'])

        model._eval()

        pbar = tqdm(total=len(loader_all))

        preds = []
        with torch.no_grad():
            for (xb, _) in loader_all:
                if model.use_cuda:
                    xb = xb.cuda()
                pred = model.cls(xb)
                preds.append(pred)
                pbar.update(1)
        preds = torch.cat(preds, dim=0).cpu().numpy()
        with open("%s/dump_preds.pkl" % expt_dir, "wb") as f:
            pickle.dump(preds, f)
