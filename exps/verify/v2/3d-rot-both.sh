#!/bin/bash

# This is 3d-rotboth.
# **verified with notebook**
declare -a arr=("s3/ckv2-mrt-3d-rotboth-best-rmxbv"
"s1/ckv2-mrt-3d-rotboth-best-rmxbv"
"s2/ckv2-mrt-3d-rotboth-best-rmxbv"
"s5/ckv2-mrt-3d-rotboth-best-rmxbv"
"s4/ckv2-mrt-3d-rotboth-best-rmxbv"
"s0/ckv2-mrt-3d-rotboth-mbrjp")
declare -a ids=(1641961 1641959 1641960 1641963 1641962 1641736)
declare -a epochs=(100 100 100 100 100 100)
