import os
import argparse

import time
import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import remote

from ppr import forward_push_single, forward_push_batch
from utils import get_data_path
from graph import GraphShard

RUNS = 1
WARMUP = 0

parser = argparse.ArgumentParser()
parser.add_argument('--num_machine', type=int, default=4, help='number of machines (simulated as processes)')
parser.add_argument('--num_roots', type=int, default=10, help='number of source nodes in each machine')
parser.add_argument('--alpha', type=float, default=0.462, help='teleport probability')
parser.add_argument('--epsilon', type=float, default=1e-6, help='maximum residual')
parser.add_argument('--version', type=str, default='single', help='version of PPR implementation')
parser.add_argument('--worker_name', type=str, default='worker{}', help='name of workers, formatted by rank')
parser.add_argument('--file_path', type=str, default='', help='path to dataset')
parser.add_argument('--log', action='store_true', help='whether to log breakdown runtime')


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

        ppr_func_dict = {
            'single': forward_push_single,
            'batch': forward_push_batch,
        }

        tik_ = time.perf_counter()
        for i in range(RUNS + WARMUP):
            if i == WARMUP:
                tik_ = time.perf_counter()

            tik = time.perf_counter()

            ppr_func_dict[args.version](rrefs, args.num_roots, args.alpha, args.epsilon)

            # futs = []
            # for rref in rrefs:
            #     futs.append(
            #         rpc.rpc_async(
            #             rref.owner(),
            #             ppr_func_dict[args.version],
            #             args=(rrefs, args.num_roots, args.alpha, args.epsilon)
            #         )
            #     )
            # c = []
            # for fut in futs:
            #     c.append(fut.wait())

            tok = time.perf_counter()
            print(f'Run {i}, Time = {tok - tik:.3f}s')

        tok_ = time.perf_counter()
        print(f'Avg Execution time = {(tok_ - tik_)/RUNS:.3f}s')

    rpc.shutdown()


if __name__ == '__main__':
    args = parser.parse_args()
    if len(args.file_path) == 0:
        args.file_path = os.path.join(get_data_path(), 'hz-ogbn-product-p{}'.format(args.num_machine))

    print('Spawn Multi-Process to simulate Multi-Machine scenario')
    start = time.time()
    mp.spawn(run, nprocs=args.num_machine, args=(args,), join=True)
    end = time.time()

    print(f'Total Execution time = {end - start:.3}s')
