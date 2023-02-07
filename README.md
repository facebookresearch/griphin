# Graph Engine

## Installation
- Compile C++ engine
    ```
    python engine/setup.py build_ext --build-lib=frontend
    ```
  NOTE: if you previously compiled with pip, you need to uninstall the pip package to avoid conflict.

## Run Examples

### Graph Engine Impl

- Distributed random walk 
  ```
  python frontend/run_rw.py --num_root 8192 --walk_length 15 --num_machine 4
  ```
- Distributed PPR
  ```
  python frontend/run_ppr.py
  ```

### PYG Baseline

- Distributed random walk
  ```
  python example/python_randomwalk/main.py
  ```
- Single Machine PPR
  ```
  python example/python_ppr/single_machine.py --max_degree 1000 --drop_coe 0.9 --num_source 100
  ```
- Distributed PPR
  ```
  python example/python_ppr/main.py
  ```