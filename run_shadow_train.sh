# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASET="ogbn-products"
MACHINES=2
DATA_PATH="data/${DATASET}-p${MACHINES}"
DGL_DATASET_PATH="/data/gangda/dgl"
FILE_PATH="intermediate"

DEVICE=0
RUNS=5
EPOCHS=35
BATCH_SIZE=128

MODEL="GAT-NORM-ACT"
DIM=256
POOL="max"


python shadow_gnn/run_shadow.py --data_name "${DATASET}" \
                                            --data_path "${DATA_PATH}" --file_path "${FILE_PATH}"\
                                            --dgl_path "${DGL_DATASET_PATH}" \
                                            --device "${DEVICE}" \
                                            --runs "${RUNS}" --epochs "${EPOCHS}" \
                                            --batch_size "${BATCH_SIZE}" \
                                            --model "${MODEL}" --dim "${DIM}" \
                                            --stand --norm \
                                            --pool "${POOL}" --feat_aug --cs

