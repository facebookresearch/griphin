#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import time

import torch
import torch.multiprocessing as mp

from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array


def foo(rank, x):
    print(x*x)
    time.sleep(1)
    return x * x


def print_shm(rank):
    # time.sleep(rank)
    t1 = get_shared_mem_array('t1', (10,), dtype=torch.int32)

    time.sleep(rank)
    print(rank, t1)
    t1[:] = torch.randperm(10, dtype=torch.int32)
    print(rank, t1)


def start_batch_p(num_process):
    t1 = create_shared_mem_array('t1', (10,), dtype=torch.int32)
    # t1 = torch.zeros(10, dtype=torch.int32).share_memory_()
    # t1[:] = torch.randperm(10, dtype=torch.int32)
    print(t1.is_shared(), '\n')

    with mp.Pool(num_process) as pool:
        futs = [pool.apply_async(print_shm, args=(i, )) for i in range(num_process)]
        results = [fut.get() for fut in futs]


def test_batching():
    num_source = 11
    num_process = 5
    num_data = int(num_source / num_process)
    # assert num_source % num_process == 0

    for i in range(num_process-1):
        print(i * num_data, (i + 1) * num_data)

    print((num_process-1)*num_data, num_source)


if __name__ == '__main__':
    start_batch_p(5)
    # mp.spawn(foo, nprocs=10, args=(1,), join=True)
    # test_batching()

