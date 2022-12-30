#!/bin/bash

# 3d
declare -a arr=(
"s0/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
"s1/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
"s2/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
"s3/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
"s4/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
"s5/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
)
#declare -a epochs=(60 60 60 60 60 60)
# SORRY: lost some checkpoints...
declare -a epochs=(183 180 184 182 60 60)
declare -a ids=(1531927 1531928 1531929 1531930 1531931 1531932)