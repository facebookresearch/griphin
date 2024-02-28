# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

ALPHA=0.261
EPSILON=1e-05
K=150

MACHINES=2  # number of machines(graph partitions)
PPR_PROCESSES=20  # number of processes used for PPR calculation per machine
EGO_PROCESSES=20  # number of processes used for EGO-Graph construction

DATASET="ogbn-products"
DGL_DATASET_PATH="/data/gangda/dgl"
DATA_PATH="data/${DATASET}-p${MACHINES}"
FILE_PATH="test_dir"

PPR_FILE="${FILE_PATH}/${DATASET}_${ALPHA}_${EPSILON}_top${K}.pt"
EGO_GRAPH_FILE="${FILE_PATH}/${DATASET}_egograph_datas.pt"


# calculate full-graph SSPPR
python frontend/run_ppr.py --alpha "${ALPHA}" --epsilon "${EPSILON}" --k "${K}"\
                           --dataset "${DATASET}" \
                           --data_path "${DATA_PATH}" \
                           --inference_out_path "${FILE_PATH}" \
                           --num_machines "${MACHINES}" \
                           --num_processes "${PPR_PROCESSES}" \
                           --num_threads 4 \
                           --version overlap \
                           --global_nid

# construct full-graph Top-K PPR matrix (deprecated)
#python shadow_gnn/pprmatrix.py --K "${K}" \
#                               --dataset "${DATASET}" \
#                               --data_path "${DATA_PATH}" \
#                               --file_path "${FILE_PATH}" \
#                               --num_machines "${MACHINES}" \
#                               --num_processes "${PPR_PROCESSES}"

# extract ego-graphs for every nodes
python shadow_gnn/egograph.py --data_name "${DATASET}" \
                              --dgl_path "${DGL_DATASET_PATH}" \
                              --data_path "${DATA_PATH}" \
                              --ppr_file "${PPR_FILE}" \
                              --num_processes "${EGO_PROCESSES}"

# load ego-graph datas into shared memory (140GB)
python shadow_gnn/load_shm.py --datas_file "${EGO_GRAPH_FILE}"
