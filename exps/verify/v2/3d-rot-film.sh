#!/bin/bash


# TODO: why does it not perform as well? I feel like
# some change in the code has messed with the eval numbers
# somehow.
declare -a arr=("s3/ckv2-mrt-3d-rotfilm-vscru"
"s1/ckv2-mrt-3d-rotfilm-vscru"
"s2/ckv2-mrt-3d-rotfilm-vscru"
"s5/ckv2-mrt-3d-rotfilm-vscru"
"s4/ckv2-mrt-3d-rotfilm-vscru"
"s0/ckv2-mrt-3d-rotfilm-dmaar")
declare -a ids=(1642877 1642873 1642875 1642881 1642879 1641703)
declare -a epochs=(60 60 60 60 60 60)
