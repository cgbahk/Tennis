#!/bin/bash
#
# Minimal preparation for this script:
# - data/
#   - annotations/labels/
#     - {vname}.txt
#   - (frames/{vname}.mp4/): optional, generated if not exist
#   - splits/minimal/
#     - default.txt
#   - videos/
#     - {vname}.mp4
#
# NOTE This doesn't require features
# NOTE Make feature with option `--save_feats`

set -e

PYTHON_OPT=""

if [ "${ENABLE_PUDB}" = 1 ]; then
  PYTHON_OPT="${PYTHON_OPT} -m pudb"
fi

python ${PYTHON_OPT} evaluate.py \
  --data_root_dir local/data.bmt \
  --num_gpus 0 \
  --video_path local/result.mp4 \
  --split_id minimal \
  --split default \
  --model_id 0006 \
  --backbone DenseNet121
