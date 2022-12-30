#!/bin/bash

RND_ID=`python rnd_id.py`

cd ..

NUM_WORKERS=4

SAVEDIR_BASE=/mnt/public/results/beckhamc/holo_ae/2022
NAME="eval-launcher-v2"
SAVEDIR=${SAVEDIR_BASE}/${NAME}-${RND_ID}
echo $SAVEDIR
mkdir $SAVEDIR

python task_launcher_eval.py \
--json_file $1 \
-d /mnt/public/datasets \
-sb ${SAVEDIR} \
--num_workers ${NUM_WORKERS}  \
-r 1 \
-j 1 \
-c "chris_v2"
