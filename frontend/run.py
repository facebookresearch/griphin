import os
import argparse
import time

import time
import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote
from contextlib import nullcontext

from utils import get_root_path
from graph import GraphShard
from random_walk import random_walk

# NUM_MACHINES = 4
# NUM_ROOTS = 8192
# WALK_LENGTH = 15
# WORKER_NAME = 'worker{}'
# FILE_PATH = 'engine/ogbn_files_txt_small'

default_file_path = os.path.join(get_root_path(), 'engine/ogbn_files_txt_small')

parser = argparse.ArgumentParser()
parser.add_argument('--num_machine', type=int, default=4, help='number of machines (simulated as processes)')
parser.add_argument('--num_root', type=int, default=8192, help='number of root nodes in each machine')
parser.add_argument('--walk_length', type=int, default=15, help='walk length')
parser.add_argument('--worker_name', type=str, default='worker{}', help='name of workers, formatted by rank')
parser.add_argument('--file_path', type=str, default=default_file_path, help='path to dataset')
parser.add_argument('--profile', action='store_true', help='whether to use torch.profile to profile program. '
                                                           'Note: this will create overheads and slow down program.')
parser.add_argument('--profile_prefix', type=str, default='tb_log/', help='path to profiling log')


def run(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=4)

    rpc.init_rpc(args.worker_name.format(rank), rank=rank, world_size=args.num_machine, rpc_backend_options=options)

    if rank == 0:
        rrefs = []
        for machine_rank in range(args.num_machine):
            info = rpc.get_worker_info(args.worker_name.format(machine_rank))
            rrefs.append(remote(info, GraphShard, args=(args.file_path, machine_rank)))

        tik_ = time.time()

        futs = []
        for rref in rrefs:
            futs.append(
                rpc.rpc_async(
                    rref.owner(),
                    random_walk,
                    args=(rrefs, args.num_machine, args.num_root, args.walk_length, args.profile,
                          '{}/{}'.format(args.profile_prefix, rref.owner()))
                )
            )

        c = []
        for fut in futs:
            c.append(fut.wait())
        tok_ = time.time()

        print(f'Random walk summary:\n {torch.cat(c, dim=0)}')
        print(f'Inner Execution time = {tok_ - tik_:.3}s')

    rpc.shutdown()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.profile:
        args.profile_prefix = '{}/{}'.format(args.profile_prefix, time.time())

    print('Spawn Multi-Process to simulate Multi-Machine scenario')
    tik = time.time()
    mp.spawn(run, nprocs=args.num_machine, args=(args,), join=True)
    tok = time.time()

    print(f'Outer Execution time = {tok - tik:.3}s')
