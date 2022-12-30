#!/bin/bash

# This is baseline1-camera
# **verified with notebook**
declare -a arr=("s3/ckv2-mrt-baseline1-cam-enabled-fipvu"
"s1/ckv2-mrt-baseline1-cam-enabled-fipvu"
"s2/ckv2-mrt-baseline1-cam-enabled-fipvu"
"s5/ckv2-mrt-baseline1-cam-enabled-fipvu"
"s4/ckv2-mrt-baseline1-cam-enabled-fipvu"
"s0/ckv2-mrt-baseline1-cam-enabled-hugnu")
declare -a ids=(1641752 1641750 1641751 1641754 1641753 1641693)
declare -a epochs=(100 100 100 100 100 60)
