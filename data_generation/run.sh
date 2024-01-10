# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASETS=("ogbn-products" "twitter" "friendster" "ogbn-papers100M")
PARTITIONS=(2 4 8)

length1=${#DATASETS[@]}
length2=${#PARTITIONS[@]}

for ((i=0;i<length1;i++))
do
  for ((j=0;j<length2;j++))
  do
    echo "DATASET:" "${DATASETS[$i]}"  "#PARTITIONS:" "${PARTITIONS[$j]}"
    python data_generation/gen_engine_data.py --num_partition "${PARTITIONS[$j]}" \
                              --data_path /data/gangda/graph_engine \
                              --path data/"${DATASETS[$i]}"-p"${PARTITIONS[$j]}" \
                              --data "${DATASETS[$i]}"
  done
done