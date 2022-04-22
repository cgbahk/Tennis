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
#   - classes
#
# NOTE This doesn't require features
# NOTE Make feature with option `--save_feats`

set -e

DATA_ROOT_DIR=$1; shift

PYTHON_OPT=""

if [ "${ENABLE_PUDB}" = 1 ]; then
  PYTHON_OPT="${PYTHON_OPT} -m pudb"
fi

python ${PYTHON_OPT} evaluate.py \
  --data_root_dir ${DATA_ROOT_DIR} \
  --num_gpus 0 \
  --video_path local/result.mp4 \
  --split_id minimal \
  --split default \
  --model_id 0006 \
  --backbone DenseNet121
