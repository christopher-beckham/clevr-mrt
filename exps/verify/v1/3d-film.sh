#!/bin/bash

declare -a arr=(
"s0/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fixeval-karot"
"s1/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fixeval-karot"
"s2/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fixeval-karot"
"s3/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fixeval-nqbvj"
"s4/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fixeval-nqbvj"
"s5/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fixeval-nqbvj"
)
declare -a ids=(1533003 1533004 1533005 1533050 1533052 1533054)
# declare -a epochs=(60 60 60 60 60 60)
# SORRY: lost some checkpoints...
# 1533003:60.pkl not found, so change 60 -> 180
# 1533004:60.pkl not found, so change 60 -> 179
# 1533005:60.pkl not found, so change 60 -> 179
# 1533050.60.pkl not found, so change 60 -> 185
declare -a epochs=(180 179 179 185 60 60)
