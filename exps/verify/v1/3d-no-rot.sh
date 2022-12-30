#!/bin/bash

# 3d-no-rot
# **Verified**
# Only one of these exists, I have no idea where these have been
declare -a arr=(
"s0/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-disablerot-pabhj"
"s1/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-itzhw"
"s2/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-itzhw"
)
#declare -a epochs=(50 130 130)
# SORRY: lost some of these checkpoints...
declare -a epochs=(52 130 131)
declare -a ids=(1512316 1514632 1514633)
