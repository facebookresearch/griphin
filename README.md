# Griphin: Graph distRIbuted PytorcH engINe

Gangda Deng, Ömer Faruk Akgül, Hongkuan Zhou, Hanqing Zeng, Yinglong Xia, Jianbo Li, Viktor Prasanna

Contact: Gangda Deng (gangdade@usc.edu)

[Paper](https://dl.acm.org/doi/abs/10.1145/3624062.3624169)

## Project Structure
The directory structure of this project is as follows:
```
.
├── baseline                      <- Multi-machine baselines, implemented 
│   │                                in pytorch only
│   ├── python_ppr                
│   │   ├── data.py               <- Data preprocessing
│   │   ├── main.py               <- Multi-machine ppr baseline
│   │   └── single_machine.py     <- Single-machine ppr experiments
│   └── python_randomwalk
├── data_generation
│   └── gen_engine_data.py
├── engine                        <- C++ graph engine backend
│   ├── Graph.h
│   ├── PPR.h
│   ├── global.h                  <- Data type definitions, should be aligned
│   │                                with definitions in frontend/graph.py
│   └── ...
├── frontend                      <- Python graph engine frontend
│   ├── graph.py                  <- Data type definitions and wrapper classes
│   ├── ppr.py                    <- Distributed ppr workflow
│   ├── random_walk.py            <- Distributed random_walk workflow
│   ├── run_ppr.py                <- Distributed ppr entrance
│   ├── run_rw.py                 <- Distributed random_walk entrance
│   └── ...
├── shadow_gnn                    <- ShaDow-GNN example
│   ├── run_shadow.py
│   └── ...
├── test
│   ├── communication_libs        <- Test sets of torch.rpc and torch.gloo
│   └── ...
├── run_ppr.sh                    <- Graph engine experiments
├── run_shadow_preproc.sh         <- Ego-graphs preprocessing for ShaDow-GNN 
└── run_shadow_train.sh           <- Shadow-GNN training scripts
```

## Installation
### Dependencies
- [pytorch](https://pytorch.org/get-started/locally/) 1.13.1+
- [dgl](https://www.dgl.ai/pages/start.html) 1.0.0+
### Compile C++ engine
```
python engine/setup.py build_ext --build-lib=frontend
```

## Data Generation
Before running the graph engine in each machine, we need to generate the graph shards first:
```
python data_generation/gen_engine_data.py --data ogbn-products --num_partition 4 --dgl_data_path <path-to-store-dgl-dataset> --path <path-to-store-graph-engine-data>
```
Here we download and preprocess the dataset in dgl format, then convert it to our graph engine format.

## Run Examples

### Griphin Use Case
Griphin currently supports high-performance distributed `Random Walk` and `Personalized PageRank (PPR)` calculation with pytorch Tensor interface.

- Distributed random walk 
  ```
  python frontend/run_rw.py --num_roots 8192 --walk_length 15 --num_machines 4
  ```
  Key arguments: 
  - `--version 2`: version 2 is the latest and the best version
  - `--num_threads 1`: currently, set num threads to 1 gives the best result
- Distributed PPR ([Pseudocode](https://hydrapse.notion.site/Dist-PPR-Pseudocode-20761fc2a93f431ba0eb5de9478ebd40))
  ```
  python frontend/run_ppr.py --inference_out_path test_dir --alpha 0.462 --epsilon 1e-6 --k 150
  ```
  Key arguments:
  - `--inference_out_path your_path`: output path for PPR results; **perform full graph SSPPR inference if it exists**
  - `--version overlap`: `overlap` is the best version which overlaps the remote and local operations.
  `cpp_batch` presents a clearer breakdown by avoiding overlapping
  - `--num_machines 4`: number of machines (simulated as processes in a single machine) 
  - `--num_processes 1`: number of processes used for SSPPR computation per machine. Note that the computation 
    between different source nodes is embrassingly parallel. With sufficient bandwidth, the throughput is
    almost linearly proportional to the number of processes
  - `--num_threads 1`: number of threads used in c++ push (implemented with [parallel-hashmap](https://github.com/greg7mdp/parallel-hashmap))

### PyG Baseline
We provide baseline demos, which are implemented by PyG operators, for comparison.
- Distributed random walk
  ```
  python baseline/python_randomwalk/main.py
  ```
- Single Machine PPR
  ```
  python baseline/python_ppr/single_machine.py --max_degree 1000 --drop_coe 0.9 --num_source 100
  ```
- Distributed PPR
  ```
  python baseline/python_ppr/main.py
  ```

### ShaDow-GNN example
In this repo, we also provide an implementation of ShaDow-GAT with our PPR subgraph sampler.
We achieve `0.8297 +- 0.0041` test accuracy on ogbn-product dataset,
which reproduces the highest result of [ShaDow_GNN](https://github.com/facebookresearch/shaDow_GNN/tree/045b85ed09a45a8e3ef58b26b5ac2274d0ee49b4).

For a quick start, modify the necessary parts in the following shell scripts and start running.
  ```
  sh run_shadow_preproc.sh
  sh run_shadow_train.sh
  ```

To eliminate the data loading time before training, we first load the subgraph data into the shared
memory once and then conduct multiple trials without data loading.
  ```
  python shadow_gnn/load_shm.py --datas_file "${FILE_PATH}/${DATASET}_egograph_datas.pt"
  sh run_shadow_train.sh
  ```

## License
Griphin is MIT licensed, as found in the LICENSE file.

## Citation
```
@inproceedings{deng2023efficient,
  title={An Efficient Distributed Graph Engine for Deep Learning on Graphs},
  author={Deng, Gangda and Akg{\"u}l, {\"O}mer Faruk and Zhou, Hongkuan and Zeng, Hanqing and Xia, Yinglong and Li, Jianbo and Prasanna, Viktor},
  booktitle={Proceedings of the SC'23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis},
  pages={922--931},
  year={2023}
}
```