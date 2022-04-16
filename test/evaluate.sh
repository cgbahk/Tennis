#!/bin/bash
#
# Minimal preparation for this script:
# - data/
#   - annotations/labels/
#     - V006.txt
#   - features/0006/V0006.mp4/
#     - ...
#   - frames/V006.mp4/
#     - ...
#   - splits/minimal/
#     - default.txt
#   - videos/
#     - V006.mp4
#
# TODO Generate features for custom video

set -e

PYTHON_OPT=""

if [ "${ENABLE_PUDB}" = 1 ]; then
  PYTHON_OPT="${PYTHON_OPT} -m pudb"
fi

python ${PYTHON_OPT} evaluate.py \
  --data_root_dir local/data \
  --num_gpus 0 \
  --video_path local/result.mp4 \
  --split_id minimal \
  --split default \
  --model_id 0042 \
  --backbone DenseNet121 \
  --temp_pool gru \
  --window 30 \
  --backbone_from_id 0006 \
  --feats_model 0006 \
  --freeze_backbone
