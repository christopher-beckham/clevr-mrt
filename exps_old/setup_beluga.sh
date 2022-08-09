#!/bin/bash

# BELUGA
module load python/3.7.0
module load cuda/10.0.130

source ~/pytorch-env/bin/activate

FLOW_DIR=/scratch/cjb60/github/flow/flow/bin
export FLOW_DIR
alias flow-submit="python $FLOW_DIR/submit.py"
