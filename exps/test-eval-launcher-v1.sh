#!/bin/bash

source env_v1.sh

cd ..

NUM_WORKERS=4

SAVEDIR_BASE=/mnt/public/results/beckhamc/holo_ae/2022
NAME="test-eval-launcher-v1"
SAVEDIR=${SAVEDIR_BASE}/${NAME}
echo $SAVEDIR
mkdir $SAVEDIR

python task_launcher_eval.py \
--json_file $1 \
-d /mnt/public/datasets \
-sb ${SAVEDIR} \
--num_workers ${NUM_WORKERS}  \
-r 1
#-j 1
