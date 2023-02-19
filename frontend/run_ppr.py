import os
import argparse

import time
import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import remote

from ppr import cpp_push_single, cpp_push_batch, python_push_single, python_push_batch, local_push
from utils import get_data_path
from graph import GraphShard, VERTEX_ID_TYPE

from graph_engine import Graph

RUNS = 10
WARMUP = 3

parser = argparse.ArgumentParser()
parser.add_argument('--num_machine', type=int, default=4, help='number of machines (simulated as processes)')
parser.add_argument('--num_process', type=int, default=1, help='number of processes in each machine used for SSPPR computing')
parser.add_argument('--num_root', type=int, default=10, help='number of source nodes in each machine')
parser.add_argument('--alpha', type=float, default=0.462, help='teleport probability')
parser.add_argument('--epsilon', type=float, default=1e-6, help='maximum residual')
parser.add_argument('--num_threads', type=int, default=4, help='num of threads to create in push operation')
parser.add_argument('--version', type=str, default='cpp_batch', help='version of PPR implementation')
parser.add_argument('--worker_name', type=str, default='worker{}', help='name of workers, formatted by rank')
parser.add_argument('--file_path', type=str, default='', help='path to dataset')
parser.add_argument('--log', action='store_true', help='whether to log breakdown runtime')


def run(rank, args, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29503'

    rpc.init_rpc(
        args.worker_name.format(rank),
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=8)
    )

    if rank == 0:

        ppr_func_dict = {
            'cpp_single': cpp_push_single,
            'cpp_batch': cpp_push_batch,
            'python_single': python_push_single,
            'python_batch': python_push_batch,
        }

        # initialize Graph Tensors in shared memory

        rrefs = []
        # for rank_ in range(0, world_size):
        for rank_ in range(0, args.num_machine):
            if rank_ < args.num_machine:
                # worker 0 ~ num_machine-1 served for remote fetch
                machine_rank = rank_
            else:
                # worker num_machine ~ world_size-1 served for local fetch
                machine_rank = int((rank_ - args.num_machine) / args.num_process)
            info = rpc.get_worker_info(args.worker_name.format(rank_))
            # rrefs.append(remote(info, GraphShard, args=(machine_rank, args.file_path)))
            rrefs.append(remote(info, Graph, args=(machine_rank, args.file_path)))

        time_pop = 0
        time_local_fetch = 0
        time_remote_fetch = 0
        time_push = 0


        tik_ = time.perf_counter()
        for i in range(RUNS + WARMUP):
            if i == WARMUP:
                tik_ = time.perf_counter()

            tik = time.perf_counter()

            source_nodes = torch.arange(args.num_root, dtype=VERTEX_ID_TYPE)

            num_data = int(args.num_root / args.num_process)

            futs = []
            for machine_rank in range(0, 1):
            # for machine_rank in range(0, args.num_machine):
                for process_rank in range(0, args.num_process):
                    emit_rank = args.num_machine + machine_rank * args.num_process + process_rank
                    # emit_rank = machine_rank
                    # data slice
                    start = process_rank * num_data
                    end = args.num_root if process_rank == args.num_process - 1 else (process_rank + 1) * num_data
                    print(f'emit_rank {emit_rank}, machine_rank {machine_rank}, start {start}, end {end}')
                    futs.append(
                        rpc.rpc_async(
                            args.worker_name.format(emit_rank),
                            local_push,
                            args=(machine_rank, process_rank, rrefs, args.num_machine,
                                  source_nodes[0:10], args.alpha, args.epsilon, args.log)
                        )
                    )

            c = []
            for j, fut in enumerate(futs):
                res = fut.wait()
                c.append(res)
                if j == 0 and i >= WARMUP:
                    time_pop += res[1]
                    time_local_fetch += res[2]
                    time_remote_fetch += res[3]
                    time_push += res[4]

            # futs = []
            # for rref in rrefs:
            #     futs.append(
            #         rpc.rpc_async(
            #             rref.owner(),
            #             ppr_func_dict[args.version],
            #             args=(rrefs, args.num_root, args.alpha, args.epsilon, args.log)
            #         )
            #     )

            tok = time.perf_counter()
            print(f'Run {i}, Time = {tok - tik:.3f}s\n')

        print(f'Avg Time pop: {time_pop / RUNS:.3f}s, '
              f'Avg Time fetch local: {time_local_fetch / RUNS:.3f}s, '
              f'Avg Time fetch remote: {time_remote_fetch / RUNS:.3f}s, '
              f'Avg Time push: {time_push / RUNS:.3f}s')

        tok_ = time.perf_counter()
        print(f'Avg Execution time = {(tok_ - tik_)/RUNS:.3f}s')

    rpc.shutdown()


if __name__ == '__main__':
    args = parser.parse_args()
    if len(args.file_path) == 0:
        args.file_path = os.path.join(get_data_path(), 'hz-ogbn-product-p{}'.format(args.num_machine))

    # world_size = args.num_machine * (args.num_process + 1)
    # world_size = args.num_machine
    world_size = args.num_machine + args.num_process

    print(f'Spawn {world_size}-Process to simulate {args.num_machine}-Machine scenario')
    t1 = time.time()

    # processes = []
    # for rank in range(world_size):
    #     p = mp.Process(target=run, args=(args, world_size))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    mp.spawn(run, nprocs=world_size, args=(args, world_size), join=True)

    t2 = time.time()
    print(f'Total Execution time = {t2 - t1:.3f}s')
