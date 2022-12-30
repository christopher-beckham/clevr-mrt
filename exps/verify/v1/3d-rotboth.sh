#!/bin/bash

# 3d-rot-via-both
declare -a arr=("s0/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-nlfpx"
                "s1/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-fixeval-mgnoo"
                "s2/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-fixeval-mgnoo"
                "s3/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-fixeval-mgnoo"
                "s4/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-fixeval-mgnoo"
                "s5/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-fixeval-mgnoo"
                )
declare -a epochs=(60 60 60 60 60 60)
declare -a ids=(1546614
                1558333
                1558334
                1558335
                1558336
                1558337)
