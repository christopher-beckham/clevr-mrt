#!/bin/bash

RESULTS_DIR="/results/holo_ae"
COPY_TO="/host_home/model_backup"

for seed in s0 s1 s2 s3 s4 s5 s6 s7; do

while read expname; do
  if [ -d $RESULTS_DIR/$seed/$expname ]; then
    echo "Found: " "$RESULTS_DIR/$seed/$expname"
    if [ ! -d $COPY_TO/$seed ]; then
      mkdir $COPY_TO/$seed
    fi
    cp -r $RESULTS_DIR/$seed/$expname $COPY_TO/$seed/$expname
  fi
  #echo "  folder size: " `du -h $RESULTS_DIR/s0/$expname`
done <backup.txt

done
