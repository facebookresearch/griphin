# Griphin: Graph distRIbuted PytorcH engINe

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
- [pytorch](https://pytorch.org/get-started/locally/) 1.13.1
- [dgl](https://www.dgl.ai/pages/start.html) 1.0.0
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

### Graph Engine Impl
- Distributed random walk 
  ```
  python frontend/run_rw.py --num_roots 8192 --walk_length 15 --num_machines 4
  ```
  Default arguments: 
  - `--version 2`: version 2 is latest and best version
  - `--num_threads 1`: currently, set num threads to 1 gives the best result
- Distributed PPR ([Pseudocode](https://hydrapse.notion.site/Dist-PPR-Pseudocode-20761fc2a93f431ba0eb5de9478ebd40))
  ```
  python frontend/run_ppr.py --num_roots 10 --alpha 0.462 --epsilon 1e-6 --num_machines 4
  ```
  Default arguments: 
  - `--version cpp_batch`: `cpp_batch` is the latest version with clear runtime breakdown.
 `overlap` is the best version which overlaps the remote and local operations in `cpp_batch`.   
  - `--num_processes 1`: number of processes used for SSPPR computation. 
  Due to a [bug in torch.rpc](https://github.com/metaopt/torchopt/issues/96), we can only set 
  num_processes to a maximum of 3.
  - `--num_threads 1`: number of threads used in c++ push

### PYG Baseline
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

### Shadow-GNN example
In this repo, we also provide an implementation of ShaDow-GAT with PPR subgraph sampler.
We are able to achieve a test accuracy of 0.8317 on the ogbn-product dataset,
which fully reproduces the highest result of ShaDow's repo.

For a quick start, modify the necessary parts in the following shell scripts and start running.
  ```
  sh run_shadow_preproc.sh
  sh run_shadow_train.sh
  ```

To eliminate the data loading time before training, we first load the subgraph data into shared
memory once and then conduct multiple trials without data loading.
  ```
  python shadow_gnn/load_shm.py --datas_file "${FILE_PATH}/${DATASET}_egograph_datas.pt"
  sh run_shadow_train.sh
  ```

## License
Griphin is MIT licensed, as found in the LICENSE file.
