#!/bin/bash

set -e

python train.py \
  --image-dir /data/hxly/datasets/BoneAge/RSNA/rsna_all \
  --train-csv /data/hxly/datasets/BoneAge/RSNA/annotations/RSNA_Annotations/RSNA_Boneage_Training.csv \
  --val-csv /data/hxly/datasets/BoneAge/RSNA/annotations/RSNA_Annotations/RSNA_Boneage_Validation.csv \
  --output-dir outputs/rsna_clip_baseline
