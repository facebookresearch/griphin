# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASETS=("ogbn-products" "twitter" "friendster" "ogbn-papers100M")
PROCESSES=(1)
THREADS=4

MACHINES=4
ROOTS=32

length1=${#DATASETS[@]}
length2=${#PROCESSES[@]}


for ((i=0;i<$length1;i++))
do
  for ((j=0;j<$length2;j++))
  do
    echo -e
    echo "DATASET:" "${DATASETS[$i]}"  "#PROCESSES:" "${PROCESSES[$j]}"
    echo -e
    python frontend/run_ppr.py --num_machines $MACHINES \
                               --file_path data/"${DATASETS[$i]}"-p$MACHINES \
                               --num_processes "${PROCESSES[$j]}" \
                               --num_threads $THREADS \
                               --num_roots $ROOTS \
#                               --version overlap
  done
done
