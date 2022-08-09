import os
import torch
import argparse
from torch import nn
from torch import optim
from torch.utils.data import (TensorDataset,
                              DataLoader,
                              Subset)
import numpy as np
from models.regressor import Regressor

from models.base import Base
import pickle
import glob
import yaml

from tools import (generate_name_from_args)

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
    parser.add_argument('--depth', type=int,
                        default=4)
    parser.add_argument('--nf', type=int,
                        default=512)
    parser.add_argument('--sigma', type=float,
                        default=0.)
    parser.add_argument('--weight_decay', type=float,
                        default=0.)
    parser.add_argument('--optim', type=str,
                        default='adam',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float,
                        default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5,
                        help="beta1 term of ADAM")
    parser.add_argument('--beta2', type=float, default=0.999,
                        help="beta2 term of ADAM")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--resume', type=str, default='auto')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--convert', type=str, default=None)
    #
    args = parser.parse_args()
    return args

# This dictionary's keys are the ones that are used
# to auto-generate the experiment name. The values
# of those keys are tuples, the first element being
# shortened version of the key (e.g. 'dataset' -> 'ds')
# and a function which may optionally shorten the value.
id_ = lambda x: str(x)
KWARGS_FOR_NAME = {
    'arch': ('arch', lambda x: os.path.basename(x)),
    'batch_size': ('bs', id_),
    'sigma': ('sigma', id_),
    'nf': ('nf', id_),
    'depth': ('d', id_),
    'lr': ('lr', id_),
    'beta1': ('b1', id_),
    'beta2': ('b2', id_),
    'weight_decay': ('wd', id_),
    'optim': ('opt', id_),
    'trial_id': ('_trial', id_)
}

def get_network(nf):
    net = [nn.Linear(68*2, nf),
           nn.BatchNorm1d(nf),
           nn.ReLU()]
    for k in range(args['depth']):
        net += [nn.Linear(nf, nf),
                nn.BatchNorm1d(nf),
                nn.ReLU()]
    net += [nn.Linear(nf, 3),
            nn.Tanh()]

    net = nn.Sequential(*net)
    return net

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

    nf = args['nf']

    net = get_network(nf)

    print("** ARCHITECTURE **")
    print(net)

    with open("/host_home/datasets/aflw/aflw_only_68_kpts_pose.pkl", "rb") as f:
        dat = pickle.load(f, encoding='latin1')
        x_all = dat['kpts'].reshape(-1, 68*2)
        y_all = dat['pose'] #[:,2:]

    # Assign 10% to be valid.
    idcs = np.arange(len(x_all))
    rnd_state = np.random.RandomState(0)
    rnd_state.shuffle(idcs)
    train_idcs = idcs[0:int(len(idcs)*0.9)]
    valid_idcs = idcs[int(len(idcs)*0.9)::]

    x_all = (torch.from_numpy(x_all).float() - 0.5) / 0.5
    y_all = torch.from_numpy(y_all).float()

    ds_all = TensorDataset(x_all, y_all)
    ds_train = Subset(ds_all, indices=train_idcs)
    ds_valid = Subset(ds_all, indices=valid_idcs)

    loader_train = DataLoader(ds_train,
                              shuffle=True,
                              batch_size=args['batch_size'])
    loader_valid = DataLoader(ds_valid,
                              shuffle=False,
                              batch_size=args['val_batch_size'])

    if args['optim'] == 'adam':
        opt_class = optim.Adam
        opt_args = {'lr': args['lr'],
                    'betas': (args['beta1'], args['beta2']),
                    'weight_decay': args['weight_decay']}
    else:
        opt_class = optim.SGD
        opt_args = {'lr': args['lr'],
                    'weight_decay': args['weight_decay']}

    model = Regressor(cls=net,
                      opt=opt_class,
                      opt_args=opt_args,
                      sigma=args['sigma'])

    if args['resume'] is not None:
        if args['resume'] == 'auto':
            # autoresume
            model_dir = "%s" % expt_dir
            # List all the pkl files.
            files = glob.glob("%s/*.pkl" % model_dir)
            # Make them absolute paths.
            files = [os.path.abspath(key) for key in files]
            if len(files) > 0:
                # Get creation time and use that.
                latest_model = max(files, key=os.path.getctime)
                print("Auto-resume mode found latest model: %s" %
                      latest_model)
                model.load(latest_model)
        else:
            model.load(args['resume'])

    if args['convert'] is None:

        model.train(itr_train=loader_train,
                    itr_valid=loader_valid,
                    epochs=args['epochs'],
                    model_dir=expt_dir,
                    result_dir=expt_dir,
                    save_every=args['save_every'])

    else:

        # Otherwise, let's convert a pickle!

        from tqdm import tqdm

        with open(args['convert'], "rb") as f:
            #try:
            #dat = pickle.load(f)
            #except:
            #    # In case the pickle was made in python 2
            dat = pickle.load(f, encoding='latin1')
            dat = dat['kpts'].reshape(-1, 68*2)
            dat = (torch.from_numpy(dat).float() - 0.5) / 0.5
            ds_query = TensorDataset(dat)
            loader_query = DataLoader(ds_query,
                                      shuffle=False,
                                      batch_size=args['val_batch_size']) # TODO: val_batch_size
            model.cls.eval()
            pbar = tqdm(total=len(loader_query))
            preds = []
            for (xb,) in loader_query:
                if model.use_cuda:
                    xb = xb.cuda()
                with torch.no_grad():
                    preds.append(model.cls(xb).cpu().numpy())
                    pbar.update(1)
            preds = np.vstack(preds)
        file_ext = args['convert'].split(".")[-1]
        dest_file = args['convert'].replace(file_ext, ".2pose.pkl")
        print("Writing to %s ..." % dest_file)
        with open(dest_file, "wb") as f:
            pickle.dump(preds, f)
