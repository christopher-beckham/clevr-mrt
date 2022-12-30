#!/bin/bash

if [ -z $DATASET_CLEVR_VERSION ]; then
  echo "ERROR: DATASET_CLEVR_VERSION not defined, did you source env_v1.sh?"
  exit 1
else
  if [ $DATASET_CLEVR_VERSION != "v1" ]; then
    echo "Error: DATASET_CLEVR_VERSION is not v1, did you source env_v1.sh?"
  fi
fi

# This is baseline0
# **Verified in control_panel_postdeadline.ipynb**
#declare -a arr=(
#"s0/ckv4b-nocc-raw-film-layer03-no-affine-gru-wd-coordconv-baseline0-best-xlvjo"
#"s1/ckv4b-nocc-raw-film-layer03-no-affine-gru-wd-coordconv-baseline0-best-xlvjo"
#"s2/ckv4b-nocc-raw-film-layer03-no-affine-gru-wd-coordconv-baseline0-best-xlvjo"
#)
#declare -a epochs=(60 60 60)
#declare -a ids=(1547108 1547109 1547110)

######################

# This is baseline1-nocamera
# **Verified**
#declare -a arr=(
#"s0/ckv4b-nocc-raw-film-layer03-no-affine-gru-wd-coordconv-nocamera-sezrg"
#"s1/ckv4b-nocc-raw-film-layer03-no-affine-gru-wd-coordconv-nocamera-best-svhrw"
#"s2/ckv4b-nocc-raw-film-layer03-no-affine-gru-wd-coordconv-nocamera-best-svhrw"
#)
### old: (120, 60, 60) but 120.pkl does not exist for some reason,
# so change 120 -> 126
#declare -a epochs=(126 60 60)
#declare -a ids=(1493658 1505290 1505291)

######################

# This is baseline1-withcamera
# **Verified**
#declare -a arr=(
#"s0/ckv4b-nocc-raw-film-layer03-no-affine-gru-wd-coordconv-camera-best-oqutz"
#"s1/ckv4b-nocc-raw-film-layer03-no-affine-gru-wd-coordconv-camera-best-mdojj"
#"s2/ckv4b-nocc-raw-film-layer03-no-affine-gru-wd-coordconv-camera-best-mdojj"
#)
#declare -a epochs=(60 60 58)
#declare -a ids=(1536718 1534737 1534738)


# 3d
#declare -a arr=(
# **Verified**
#"s0/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
#"s1/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
#"s2/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
#"s3/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
#"s4/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
#"s5/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
#)
#declare -a epochs=(60 60 60 60 60 60)
#declare -a ids=(1531927 1531928 1531929 1531930 1531931 1531932)
#
# [0,1,2,3] are missing, why? These are the only ones that are left:
#declare -a arr=(
#"s4/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
#"s5/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fiximageneteval-yjgfg"
#)
#declare -a epochs=(60 60)
#declare -a ids=(1531931 1531932)


# 3d-no-rot
# **Verified**
# Only one of these exists, I have no idea where these have been
declare -a arr=("s0/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-disablerot-pabhj"
                "s1/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-itzhw"
                "s2/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-itzhw"
               )
#what the fuck is this one?
declare -a epochs=(50 130 130)
declare -a ids=(1512316 1514632 1514633)


# 3d-film
# **Verified**
#declare -a arr=(
#"s0/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fixeval-karot"
#"s1/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fixeval-karot"
#"s2/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fixeval-karot"
#"s3/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fixeval-nqbvj"
#"s4/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fixeval-nqbvj"
#"s5/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-best-fixeval-nqbvj"
#)
#declare -a epochs=(60 60 60 60 60 60)
#declare -a ids=(1533003 1533004 1533005 1533050 1533052 1533054)
# shit, only 2/6 have their 60.pkl still there


# 3d-rot-via-both
#declare -a arr=("s0/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-nlfpx"
#                "s1/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-fixeval-mgnoo"
#                "s2/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-fixeval-mgnoo"
#                "s3/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-fixeval-mgnoo"
#                "s4/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-fixeval-mgnoo"
#                "s5/ckv4b-holo-encoder-end2end-128px-cc-testlite3-nocc-3dlite-fixcam-imagenet-scaling-fixeval-mgnoo"
#                )
#declare -a epochs=(60 60 60 60 60 60)
#declare -a ids=(1546614
#                1558333
#                1558334
#                1558335
#                1558336
#                1558337)

cd ..

for idx in "${!arr[@]}"; do

    NAME=${arr[${idx}]}
    EPOCH=${epochs[${idx}]}
    ID=${ids[${idx}]}
    echo "Name: " $NAME
    echo "  epoch:  " $EPOCH
    echo "  id: " $ID
    echo "  pkl: " ${RESULTS_DIR}/${NAME}/${ID}/$EPOCH.pkl
    if [ -e ${RESULTS_DIR}/${NAME}/${ID}/$EPOCH.pkl ]; then
     echo "  ...exists"
    else
     echo "  ...not exist"
     #echo " but other pkl files in dir are: " `ls ${RESULTS_DIR}/${NAME}/${ID}/*.pkl`
    fi

    #python evaluate.py --experiment=${RESULTS_DIR}/${NAME}/${ID}/$EPOCH.pkl --is_legacy \
#	--dry_run

done
