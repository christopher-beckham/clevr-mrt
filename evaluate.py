import json
import argparse
import yaml
import os
import torch

from task_launcher_common import setup

from src import (job_configs,
                 exp_utils)

from src.models import setup_logger
logger = setup_logger.get_logger()

from haven import haven_utils as hu
from haven import haven_wizard as hw

from tqdm import tqdm

from src.models import setup_logger
logger = setup_logger.get_logger()

def evaluate_and_save(net, loader, out_file, verbose=False):
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
    
    if verbose:
        logger.info("Mean acc: {}".format(acc_ind.mean()))
    
    with open(out_file, "w") as f:
        f.write("correct,cam_x,cam_y,cam_z,tf,n_color_unique,n_shape_unique,n_mat_unique,n_obj\n")
        for j in range(len(cams)):
            f.write("%f,%f,%f,%f,%s,%i,%i,%i,%i\n" % \
                    (acc_ind[j], cams[j][0], cams[j][1], cams[j][2], tfs[j], nc[j], ns[j], nm[j], n_obj[j]))
    
def eval_experiment(experiment, 
                    mode, 
                    batch_size, 
                    num_workers=1, 
                    is_legacy=False, 
                    verbose=False):
    """Evaluate experiment.

    Args:
        experiment (str): Path to experiment pkl.
        mode (str): Either 'train', 'valid', or 'test'.
        batch_size (int, optional): batch size for data loaders.
            If this is None, then it will use the batch size defined
            in the cfg file of the referenced experiment. Defaults to None.
        num_workers (int, optional): number of workers for each
            data loader. Defaults to 1.
        is_legacy (bool, optional): Is this a legacy experiment?
            Set this to true if the experiment has a cfg.yaml instead of a 
            cfg.json file. Defaults to False.
    """ 
    assert mode in ['train', 'valid', 'test']
    
    exp_dir = os.path.dirname(experiment)    
    pkl_name = os.path.basename(experiment)
    
    if is_legacy:
        # First, load in the yaml file in the exp dir.
        # We can expect yaml since that was what legacy
        # experiments used.
        yaml_cfg = "{}/cfg.yaml".format(exp_dir)
        if not os.path.exists(yaml_cfg):
            raise Exception("is_legacy=True, expected yaml cfg but did not find one")
        exp_dict = yaml.load(open(yaml_cfg).read())
        logger.info("yaml -> json: {}".format(
            json.dumps(exp_dict, indent=4))
        )
        
    # Inject experiment into arch_checkpoint so that we can load
    # the chkpt in.
    #exp_dict['arch_checkpoint'] = experiment
    
    # How model loading works here:
    # - if arch_checkpoint is defined, it will load this in,
    #   otherwise, if model.pth or model_best.pth exists it
    #   will load that in instead.
    # - for contrastive probe experiments, arch_checkpoint will
    #   be defined and it will point to the contrastive encoder
    #   pkl.
    # - for regular experiments, if the experiment is legacy
    #   (which means no *.pth files), then setup() will return
    #   net but won't load any kind of checkpoint. That means we
    #   should load one in ourselves.    
    
    # setup will internally fill exp_dict and
    # validate its arguments.
    net, loaders = setup(exp_dict,
                         exp_dir,
                         batch_size=batch_size,
                         only_valid_keys=False,
                         auto_resume=True,
                         load_test=mode=='test',
                         num_workers=num_workers)
    if mode == 'train':
        loader = loaders['train']
    elif mode == 'valid':
        loader = loaders['valid']
    else:
        loader = loaders['test']
    assert loader is not None
    
    logger.info("Loading checkpoint: {}".format(experiment))
    net.load(experiment)

    evaluate_and_save(
        net, 
        loader, 
        out_file="{}/eval_{}_{}.csv".format(exp_dir, mode, pkl_name),
        verbose=verbose
    )
    
    #print(net)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default=None, required=True)
    parser.add_argument('--mode', choices=['train', 'valid'], default='valid')
    parser.add_argument('--is_legacy', action='store_true')
    parser.add_argument("--batch_size", type=int, default=32)
    
    args, others = parser.parse_known_args()
    
    eval_experiment(experiment=args.experiment,
                    mode=args.mode,
                    batch_size=args.batch_size,
                    is_legacy=args.is_legacy,
                    verbose=True)