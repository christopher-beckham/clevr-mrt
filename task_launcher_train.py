import json
import argparse

from task_launcher_common import setup

from src import (job_configs,
                 exp_utils)

from src.models import setup_logger
logger = setup_logger.get_logger()

from haven import haven_utils as hu
from haven import haven_wizard as hw

def trainval(exp_dict, savedir, args):
    
    # setup will internally fill exp_dict and
    # validate its arguments.
    net, loaders = setup(exp_dict, 
                         savedir,
                         load_test=False,
                         num_workers=args.num_workers)
    
    # HACK: haven internally saves an exp_dict.json already
    # in savedir, but we need to account for the fact that
    # we just inserted defaults into that dictionary with
    # insert_defaults. So save it again.
    with open("{}/exp_dict.json".format(savedir), "w") as f:
        logger.info("Writing {}/exp_dict.json".format(savedir))
        f.write(json.dumps(exp_dict, indent=4))  

    loader_train, loader_valid, _ = loaders
    
    if exp_dict['class'] == 'holo_encoder':
        track_metric = 'valid_probe_acc'
    else:
        track_metric = 'valid_acc'
    
    net.fit(
        itr_train=loader_train,
        itr_valid=loader_valid,
        epochs=exp_dict['epochs'],
        model_dir=savedir,
        result_dir=savedir,
        save_every=exp_dict['save_every'],
        track_metric=track_metric,
        #validate_only=exp_dict['validate_only'],
        validate_only=False,
        #debug=exp_dict['debug'],
        debug=False,
        verbose=True
    )
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+",
                        help='Define which exp groups to run.')
    parser.add_argument('--json_file', default=None)
    parser.add_argument('-sb', '--savedir_base', default=None,
                        help='Define the base directory where the experiments will be saved.')
    parser.add_argument('-d', '--datadir', default=None,
                        help='Define the dataset directory.')
    parser.add_argument("-r", "--reset",  default=0, type=int,
                        help='Reset or resume the experiment.')
    parser.add_argument("-nw", "--num_workers",  default=8, type=int,
                        help='num_workers.')
    parser.add_argument("-c", "--job_config",  default='chris', 
                        help='Choose job config name')
    args, others = parser.parse_known_args()
    
    # get job config 
    job_config = job_configs.JOB_CONFIGS[args.job_config]
    python_binary_path = 'python'

    import exp_configs

    if args.json_file not in [None, 'None']:
        # use json file to acquire experiments
        configs = exp_utils.enumerate_and_unflatten(args.json_file)
        hw.run_wizard(func=trainval,
                      exp_list=configs,
                      args=args,
                      python_binary_path=python_binary_path,
                      job_config=job_config,
                      use_threads=True)
    else:
        # use exp_config to acquire experiments
        hw.run_wizard(func=trainval,
                      exp_groups=exp_configs.EXP_GROUPS,
                      args=args,
                      python_binary_path=python_binary_path,
                      job_config=job_config,
                      use_threads=True)