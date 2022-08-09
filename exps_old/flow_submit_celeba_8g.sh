#!/bin/bash

echo "Submitting script: " $1

flow-submit \
--prolog="source ../envs/env_celeba.sh cc" \
--resume \
--options "mem=8G;time=12:00:00;account=rpp-bengioy;ntasks=1;cpus-per-task=4;gres=gpu:1" \
--root=`pwd` \
launch $1
