# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASET="ogbn-products"
DGL_DATASET_PATH="/data/gangda/dgl"
K=150

MACHINES=2  # number of machines(graph partitions)
PPR_PROCESSES=8  # number of processes used for PPR calculation per machine
EGO_PROCESSES=20  # number of processes used for EGO-Graph construction

FILE_PATH="intermediate"
DATA_PATH="data/${DATASET}-p${MACHINES}"
DATAS_FILE="${FILE_PATH}/${DATASET}_egograph_datas.pt"


# calculate full-graph SSPPR
python frontend/run_ppr.py --alpha 0.261 --epsilon 1e-5 \
                           --version overlap \
                           --dataset "${DATASET}" \
                           --data_path "${DATA_PATH}" \
                           --inference_out_path "${FILE_PATH}" \
                           --num_machines "${MACHINES}" \
                           --num_processes "${PPR_PROCESSES}" \
                           --num_threads 4 \
                           --log

# construct full-graph Top-K PPR matrix
python shadow_gnn/pprmatrix.py --K "${K}" \
                               --dataset "${DATASET}" \
                               --data_path "${DATA_PATH}" \
                               --file_path "${FILE_PATH}" \
                               --num_machines "${MACHINES}" \
                               --num_processes "${PPR_PROCESSES}"

# extract ego-graphs for every nodes
python shadow_gnn/egograph.py --data_name "${DATASET}" \
                              --dgl_path "${DGL_DATASET_PATH}" \
                              --data_path "${DATA_PATH}" \
                              --file_path "${FILE_PATH}" \
                              --num_processes "${EGO_PROCESSES}"

# load ego-graph datas into shared memory (140GB)
python shadow_gnn/load_shm.py --datas_file "${DATAS_FILE}"
