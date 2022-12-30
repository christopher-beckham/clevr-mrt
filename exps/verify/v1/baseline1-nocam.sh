#!/bin/bash

# This is baseline1-nocamera
# **Verified**
declare -a arr=(
"s0/ckv4b-nocc-raw-film-layer03-no-affine-gru-wd-coordconv-nocamera-sezrg"
"s1/ckv4b-nocc-raw-film-layer03-no-affine-gru-wd-coordconv-nocamera-best-svhrw"
"s2/ckv4b-nocc-raw-film-layer03-no-affine-gru-wd-coordconv-nocamera-best-svhrw"
)
# SORRY: should be (120, 60, 60) but 120.pkl does not exist anymore,
# so change 120 -> 126 instead.
declare -a epochs=(126 60 60)
declare -a ids=(1493658 1505290 1505291)
