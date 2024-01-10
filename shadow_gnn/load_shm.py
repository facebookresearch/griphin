#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse
import threading
import time

import torch
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array

DATA_KEYS = [
    'node_offset',
    'edge_offset',
    'sub_nidx',
    'sub_eidx',
    'ego_idx',
]
DATA_SHAPE_KEY = 'data_shapes'


def fetch_datas_from_shm():
    shm_data_shapes = get_shared_mem_array(DATA_SHAPE_KEY, (len(DATA_KEYS),), dtype=torch.long)
    datas = {}
    for i, key in enumerate(DATA_KEYS):
        shape = (2, shm_data_shapes[i]) if key == 'sub_eidx' else (shm_data_shapes[i],)
        datas[key] = get_shared_mem_array(key, shape, dtype=torch.long)
    return datas


def host_datas(e):
    print('Start Loading egograph datas')
    tik = time.time()

    datas = torch.load(args.datas_file)
    shm_data_shapes = create_shared_mem_array(DATA_SHAPE_KEY, (len(DATA_KEYS),), dtype=torch.long)
    shm_data_shapes[:] = torch.tensor([datas[key].shape[-1] for key in DATA_KEYS])

    shm_datas = []
    for key in DATA_KEYS:
        data = datas[key]
        shm_data = create_shared_mem_array(key, data.size(), dtype=torch.long)
        shm_data[:] = data
        shm_datas.append(shm_data)
    del datas

    tok = time.time()
    print(f'Loading data list to shm finished. Elapsed Time: {tok-tik:.0f}s')

    print('Press Ctrl+C to exit')
    breakpoint()
    # never reached, otherwise event.set() would be here


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datas_file',
        type=str,
        default='intermediate/ogbn-products_egograph_datas.pt'
    )
    args = parser.parse_args()

    event = threading.Event()
    threading.Thread(target=host_datas, args=[event], daemon=True).start()
    try:
        event.wait()
    except KeyboardInterrupt:
        print('Release shm')
