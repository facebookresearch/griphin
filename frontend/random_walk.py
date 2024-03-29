#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import time
from collections import OrderedDict, defaultdict
from contextlib import nullcontext

import torch
from torch.distributed import rpc

from graph import VERTEX_ID_TYPE, SHARD_ID_TYPE


def random_walk(shard_rrefs, num_machines, num_roots, walk_length, profile, profile_prefix, log=False):
    """Kernel Function of Distributed Random Walk

    :param shard_rrefs: Reference list of remote graph shards.
    :param num_machines: World size.
    :param num_roots: Number of root nodes in current machine.
    :param walk_length: Length of random walks.
    :param profile:
    :param profile_prefix:
    :param log:
    :return: walk_summary
    """

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=walk_length+1, skip_first=0, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_prefix),
        with_stack=True
    ) if profile else nullcontext() as profiler:
        rank = rpc.get_worker_info().id
        local_shard = shard_rrefs[rank].to_here()

        # init_rpc(shard_rrefs)
        # if profile:
        #     profiler.step()

        # init root node IDs (0 ~ num_core_nodes-1) of local shard
        # root_nodes = torch.arange(num_roots, dtype=VERTEX_ID_TYPE)
        root_nodes = torch.randperm(local_shard.num_core_nodes, dtype=VERTEX_ID_TYPE)[:num_roots]

        # init walks summary with size (num_roots, walk_length+1)
        # walks_summary = torch.full((num_roots, walk_length + 1), -1, dtype=root_nodes.dtype)
        walks_summary = torch.empty((num_roots, walk_length + 1), dtype=root_nodes.dtype)
        walks_summary[:, 0] = local_shard.to_global(root_nodes)

        # init node ID tensor "u" of current step
        # init index "root_perm" that maps "u" to root nodes
        u = root_nodes
        root_perm = torch.arange(num_roots)

        # init shard dict for current step "shard_dict_u" and next step "shard_dict_v"
        shard_dict_u = OrderedDict({rank: torch.arange(num_roots)})
        # shard_dict_u = OrderedDict([(i, torch.arange(num_roots))for i in range(num_machines)])
        shard_dict_v = defaultdict(list)

        if profile:
            profiler.step()

        if rank == 0:
            print(f"Starting Distributed Random Walk:"
                  f" world_size={num_machines}, num_roots={num_roots}, walk_length={walk_length}")

        part1, part2, part3 = [], [], []
        part2_1, part2_2, part2_3 = [], [], []

        for i in range(walk_length):
            """Example:
            
            Suppose we have 2 shards.
            shard_0: 
                - Core nodes: global ID (0, 1, 2, 3, 4), local ID (0, 1, 2, 3, 4)
                - Halo nodes: global ID (8, 9), local ID (3, 4)
            shard_1:
                - Core nodes: global ID (5, 6, 7, 8, 9), local ID (0, 1, 2, 3, 4)
                - Halo nodes: global ID (0, 1), local ID (0, 1)
            
            On machine 0, lets assign root nodes with global ID (0, 1, 2, 3, 4).
            Initialization:
                - u (local ID):                                [0, 1, 2, 3, 4]
                - root_perm (index that maps u to root nodes): [0, 1, 2, 3, 4]
                - walk_summary[0] (global ID):                 [0, 1, 2, 3, 4]
            Step 1:
                Given that all the nodes of current step belong to shard_0, we do not need RPC call.
                1. Invoke local_shard.sample_single_neighbor()
                    Input (local ID): [0, 1, 2, 3, 4] 
                    Return:
                        - nid (local ID): [2, 4, 3, 3, 4]
                        - shard_dict: {0: [0, 1, 2], 
                                    1: [3, 4]}
                3. Construct shard dict for next step: 
                    - shard_dict_v: {0: [0, 1, 2],
                                    1: [3, 4]}
                3. Update:
                    - root_perm:       [0, 1, 2, 3, 4]
                    - u:               [2, 4, 3, 3, 4]
                    - walk_summary[1]: [2, 4, 3, 8, 9]
            Step 2:
                1. Invoke sample_single_neighbor():
                    - shard_0:
                        Input: [2, 4, 3] 
                        Return:
                            - nid (local ID): [0, 1, 3]
                            - shard_dict: {0: [0, 1], 1: [2]}
                    - shard_1:
                        Input: [3, 4]
                        Return:
                            - nid (local ID): [2, 1]
                            - shard_dict: {0: [1], 1: [0]}
                2. Construct shard dict for next step:
                    Merge two shard_dict from shard_0 and shard_1. 
                    - shard_dict_v: {0: [0, 1, 4],
                                    1: [2, 3]}
                3. Update:
                    - root_perm:       [0, 1, 2, 3, 4]
                    - u:               [0, 1, 3, 2, 1]
                    - walk_summary[2]: [0, 1, 8, 7, 1]
            """
            shard_dict_v.clear()

            # issue async RPC call
            futs = {}
            for p, index_u in shard_dict_u.items():
                if rank == p:
                    continue
                futs[p] = shard_rrefs[p].rpc_async().sample_single_neighbor(u[index_u])

            tik_ = time.time()

            # overlap remote step with local step
            nid_local, shard_dict_local = local_shard.sample_single_neighbor(u[shard_dict_u[rank]])

            tok_ = time.time()
            part1.append(tok_ - tik_)
            tik_ = time.time()

            # collect results and constructs shard_dict_v for next step
            sum1 = sum2 = sum3 = 0
            nids_global, nids, offset = [], [], 0
            for p in shard_dict_u.keys():
                tik1_ = time.time()

                if rank == p:
                    nid, shard_dict = nid_local, shard_dict_local
                else:
                    nid, shard_dict = futs[p].wait()

                tok1_ = time.time()
                sum1 += tok1_ - tik1_
                sum2_ = 0

                nid_global = torch.empty_like(nid)
                for q, index_v in shard_dict.items():
                    tik2_ = time.time()

                    shard_dict_v[q].append(index_v + offset)  # merge shard dicts

                    tok2_ = time.time()
                    sum2_ += tok2_ - tik2_
                    tik3_ = time.time()

                    nid_global[index_v] = local_shard.to_global(nid[index_v], q)

                    tok3_ = time.time()
                    sum3 += tok3_ - tik3_

                sum2 += sum2_

                nids.append(nid)
                nids_global.append(nid_global)
                offset += nid.size(0)

            part2_1.append(sum1)
            part2_2.append(sum2)
            part2_3.append(sum3)

            tok_ = time.time()
            part2.append(tok_ - tik_)
            tik_ = time.time()

            # update root_perm, and then
            # store global IDs of target nodes in the correct order
            # (use root_perm to align target nodes with root nodes)
            perm = torch.cat(list(shard_dict_u.values()))
            root_perm = root_perm.index_select(0, perm)
            walks_summary[root_perm, i + 1] = torch.cat(nids_global)

            # update node_index & shard_dict
            # shard_dict_u must have the same shard order with rets/rets_global
            # sort shard_dict_v to preserve the order
            u = torch.cat(nids)
            shard_dict_u.clear()
            for q, index_v_list in sorted(shard_dict_v.items()):
                shard_dict_u[q] = torch.cat(index_v_list)

            tok_ = time.time()
            part3.append(tok_ - tik_)

            if profile:
                profiler.step()

    if log:
        print(f"Rank {rank} Sum for Part 1: {sum(part1):.3f} \n")
        print(f"Rank {rank} Sum for Part 2: {sum(part2):.3f} \n")
        print(f"Rank {rank} Sum for Part 3: {sum(part3):.3f} \n")
        print(f"Rank {rank} Sum for Part 2_1: {sum(part2_1):.3f} \n")
        print(f"Rank {rank} Sum for Part 2_2: {sum(part2_2):.3f} \n")
        print(f"Rank {rank} Sum for Part 2_3: {sum(part2_3):.3f} \n")
        print("****\n")

    return walks_summary


def random_walk2(shard_rrefs, num_machines, num_threads, num_roots, walk_length, profile, profile_prefix, log=False):
    """Kernel Function of Distributed Random Walk

    :param shard_rrefs: Reference list of remote graph shards.
    :param num_machines: World size.
    :param num_threads: Number of threads used in c++ sample
    :param num_roots: Number of root nodes in current machine.
    :param walk_length: Length of random walks.
    :param profile: bool
    :param profile_prefix: profile dir
    :param log: bool
    :return: walk_summary
    """

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=walk_length+1, skip_first=0, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_prefix),
        with_stack=True
    ) if profile else nullcontext() as profiler:
        rank = rpc.get_worker_info().id
        # Rank 0 ~ num_machines-1 = Server Processes; num_machines ~ 2*num_machines-1: Worker Processes
        machine_rank = rank - num_machines
        local_shard = shard_rrefs[rank].to_here()

        # init root node IDs (0 ~ num_core_nodes-1) of local shard
        # root_nodes = torch.arange(num_roots, dtype=VERTEX_ID_TYPE)
        root_nodes = torch.randperm(local_shard.num_core_nodes(), dtype=VERTEX_ID_TYPE)[:num_roots]

        # init walks summary with size (num_roots, walk_length+1)
        # walks_summary = torch.full((num_roots, walk_length + 1), -1, dtype=root_nodes.dtype)
        walks_summary = torch.empty((num_roots, walk_length + 1), dtype=root_nodes.dtype)
        walks_summary[:, 0] = root_nodes + local_shard.partition_book()[machine_rank]

        # init node ID tensor "u" of current step
        u = root_nodes

        # init shard dict for "shard_dict_u"
        shard_dict_u = OrderedDict({machine_rank: torch.arange(num_roots)})
        shard_ids = torch.empty(num_roots, dtype=SHARD_ID_TYPE)

        if profile:
            profiler.step()

        if machine_rank == 0:
            print(f"Starting Distributed Random Walk:"
                  f" total_num_nodes: {local_shard.partition_book()[-1]},"
                  f" world_size={num_machines}, num_roots={num_roots}, walk_length={walk_length}")

        part1, part2, part3 = [], [], []
        part2_1, part2_2 = [], []

        for i in range(walk_length):
            # issue async RPC call
            futs = {}
            for p, mask_p in shard_dict_u.items():
                if machine_rank == p:
                    continue
                futs[p] = shard_rrefs[p].rpc_async().sample_single_neighbor2(u[mask_p], num_threads)

            tik_ = time.time()

            # overlap remote step with local step
            local_nid_, global_nid_, shard_id_ = local_shard.sample_single_neighbor2(u[shard_dict_u[machine_rank]],
                                                                                     num_threads)

            tok_ = time.time()
            part1.append(tok_ - tik_)
            tik_ = time.time()

            # collect results and constructs shard_dict_v for next step
            sum1 = sum2 = 0
            for p in shard_dict_u.keys():
                tik1_ = time.time()

                if machine_rank == p:
                    local_nid, global_nid, shard_id = local_nid_, global_nid_, shard_id_
                else:
                    local_nid, global_nid, shard_id = futs[p].wait()

                tok1_ = time.time()
                sum1 += tok1_ - tik1_
                tik1_ = time.time()

                u[shard_dict_u[p]] = local_nid
                walks_summary[shard_dict_u[p], i+1] = global_nid
                shard_ids[shard_dict_u[p]] = shard_id

                tok1_ = time.time()
                sum2 += tok1_ - tik1_

            part2_1.append(sum1)
            part2_2.append(sum2)

            tok_ = time.time()
            part2.append(tok_ - tik_)
            tik_ = time.time()

            # update shard_dict_u
            for q in range(num_machines):
                mask = shard_ids == q
                if mask.sum() == 0:
                    shard_dict_u.pop(q, None)
                else:
                    shard_dict_u[q] = mask

            tok_ = time.time()
            part3.append(tok_ - tik_)

            if profile:
                profiler.step()

    if log:
        print(f"Rank {machine_rank} Sum for Part 1: {sum(part1):.3f} \n")
        print(f"Rank {machine_rank} Sum for Part 2: {sum(part2):.3f} \n")
        print(f"Rank {machine_rank} Sum for Part 3: {sum(part3):.3f} \n")
        print(f"Rank {machine_rank} Sum for Part 2_1: {sum(part2_1):.3f} \n")
        print(f"Rank {machine_rank} Sum for Part 2_2: {sum(part2_2):.3f} \n")
        print("****\n")

    return walks_summary
