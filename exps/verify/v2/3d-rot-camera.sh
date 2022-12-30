#!/bin/bash

# This is 3d-rot-camera
# **verified with notebook**
declare -a arr=("s3/ckv2-mrt-3d-rot-best-lxptx"
"s1/ckv2-mrt-3d-rot-best-lxptx"
"s5/ckv2-mrt-3d-rot-best-lxptx"
"s4/ckv2-mrt-3d-rot-best-lxptx"
"s0/ckv2-mrt-3d-rot-wwnxc"
"s2/ckv2-mrt-3d-rot-kceuw")
declare -a ids=(1641773 1641771 1641775 1641774 1641710 1641968)
declare -a epochs=(100 100 100 100 100 100)

