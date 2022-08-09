"""
To resume experiments which were executed with Shuriken.
"""

import subprocess
from shuriken.utils import get_hparams
import os

shk_args = get_hparams()
full_name = shk_args['full_name'] # MUST use this, in format seed/name/trial_id
if len(full_name.split("/")) != 3:
    raise Exception("full name must have 3 parts: seed/name/trial_id")
seed, name, trial_id = full_name.split("/")
seed = seed[-1] # change e.g. s0 to 0
print("trial id=" + str(trial_id))
results_dir = os.environ['RESULTS_DIR']
resume = shk_args['resume']
num_workers = shk_args['num_workers']

# HACKY: boolean arguments
if 'pin_memory' not in shk_args:
    pin_memory = ""
else:
    pin_memory = "--pin_memory"
    
#print(subprocess.run("ls -lt", shell=True))

cfg_file = "%s/%s/cfg.yaml" % (results_dir, full_name)
print("cfg file = %s" % cfg_file)

# python task_launcher.py load --config=cfg.yaml --trial_id=... --name=...

subprocess.run(["python", "task_launcher_encoder.py", "load", "--config={FILE}".format(
    FILE=cfg_file), "--trial_id", str(trial_id), "--name", name, "--resume", resume, "--num_workers", str(num_workers), pin_memory])
