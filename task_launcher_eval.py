import json
import argparse

from task_launcher_common import setup

from src import (job_configs,
                 exp_utils)

from src.models import setup_logger
logger = setup_logger.get_logger()

from haven import haven_utils as hu
from haven import haven_wizard as hw

from evaluate import eval_experiment
    
def _eval_experiment(exp_dict, savedir, args):
    eval_experiment(
        experiment=exp_dict['experiment'],
        mode=exp_dict['mode'],
        batch_size=exp_dict['batch_size'],
        is_legacy=exp_dict['is_legacy'],
        num_workers=args.num_workers,
        verbose=False
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
        hw.run_wizard(func=_eval_experiment,
                      exp_list=configs,
                      args=args,
                      python_binary_path=python_binary_path,
                      job_config=job_config,
                      use_threads=True)
    else:
        # use exp_config to acquire experiments
        hw.run_wizard(func=_eval_experiment,
                      exp_groups=exp_configs.EXP_GROUPS,
                      args=args,
                      python_binary_path=python_binary_path,
                      job_config=job_config,
                      use_threads=True)
