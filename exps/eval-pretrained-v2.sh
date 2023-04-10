#!/bin/bash

if [ -z $DATASET_CLEVR_VERSION ]; then
  echo "ERROR: DATASET_CLEVR_VERSION not defined, did you source env_v2.sh?"
  exit 1
else
  if [ $DATASET_CLEVR_VERSION != "v2" ]; then
    echo "Error: DATASET_CLEVR_VERSION is not v2, did you source env_v2.sh?"
  fi
fi


# TODO: move the bottom out into its own individual files
# source <exp name>.sh
# then copy them to their directory with checkpoints

########################
# CLEVR-MRT-V2 RESULTS #
########################

cd ..

for filename in `ls exps/verify/v2/*.sh`; do

    echo "Sourcing $filename ..."
    source $filename

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
            echo "  ...not found"
            break
        fi

        # --is_legacy is set since older experiments used yaml instead of a json
        # for config files
        #python evaluate.py --experiment=${RESULTS_DIR}/${NAME}/${ID}/$EPOCH.pkl --is_legacy \
        #    --dry_run

        # Back up the cfg file before we do anything silly
        #cp ${RESULTS_DIR}/${NAME}/${ID}/cfg.yaml ${RESULTS_DIR}/${NAME}/${ID}/cfg.yaml.bak

        # Fix the cfg file.
        #python exps/remove-old-keys.py ${RESULTS_DIR}/${NAME}/${ID}/cfg.yaml > ${RESULTS_DIR}/${NAME}/${ID}/cfg.yaml.cleaned
        #mv ${RESULTS_DIR}/${NAME}/${ID}/cfg.yaml.cleaned ${RESULTS_DIR}/${NAME}/${ID}/cfg.yaml

        #break
    done

    echo ""

done
