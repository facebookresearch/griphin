import time
from collections import OrderedDict, defaultdict
from contextlib import nullcontext

import torch
from torch.distributed import rpc

from graph import GraphShard, VERTEX_ID_TYPE, SHARD_ID_TYPE


def init_rpc(shard_rrefs):
    futs = []
    for ref in shard_rrefs:
        shard: GraphShard = ref.rpc_async()
        futs.append(shard.get_dict_tensor(torch.rand(200)))
    for fut in futs:
        fut.wait()


def random_walk2(shard_rrefs, num_machines, num_roots, walk_length, profile, profile_prefix):
    """Kernel Function of Distributed Random Walk

    :param shard_rrefs: Reference list of remote graph shards.
    :param num_machines: World size.
    :param num_roots: Number of root nodes in current machine.
    :param walk_length: Length of random walks.
    :param profile:
    :param profile_prefix:
    :return: walk_summary
    """

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=walk_length+1, skip_first=0, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_prefix),
        with_stack=True
    ) if profile else nullcontext() as profiler:
        rank = rpc.get_worker_info().id
        local_shard: GraphShard = shard_rrefs[rank].to_here()

        # init root node IDs (0 ~ num_core_nodes-1) of local shard
        # root_nodes = torch.arange(num_roots, dtype=VERTEX_ID_TYPE)
        root_nodes = torch.randperm(local_shard.num_core_nodes, dtype=VERTEX_ID_TYPE)[:num_roots]

        # init walks summary with size (num_roots, walk_length+1)
        walks_summary = torch.full((num_roots, walk_length + 1), -1, dtype=root_nodes.dtype)
        walks_summary[:, 0] = local_shard.to_global(root_nodes)

        # init node ID tensor "u" of current step
        u = root_nodes

        # init shard dict for current step "shard_dict_u" and next step "shard_dict_v"
        shard_dict_u = OrderedDict({rank: torch.arange(num_roots)})
        shard_ids = torch.empty(num_roots, dtype=SHARD_ID_TYPE)

        if profile:
            profiler.step()

        if rank == 0:
            print(f"Starting Distributed Random Walk:"
                  f" world_size={num_machines}, num_roots={num_roots}, walk_length={walk_length}")

        part1, part2, part3 = [], [], []
        part2_1, part2_2 = [], []

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
                1. Invoke local_shard.walk_one_step()
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
                1. Invoke walk_one_step():
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

            # issue async RPC call
            futs = {}
            for p, mask_p in shard_dict_u.items():
                if rank == p:
                    continue
                futs[p] = shard_rrefs[p].rpc_async().walk_one_step2(u[mask_p])

            tik_ = time.time()

            # overlap remote step with local step
            local_nid_, global_nid_, shard_id_ = local_shard.walk_one_step2(u[shard_dict_u[rank]])

            tok_ = time.time()
            part1.append(tok_ - tik_)
            tik_ = time.time()

            # collect results and constructs shard_dict_v for next step
            sum1 = sum2 = 0
            for p in shard_dict_u.keys():
                tik1_ = time.time()

                if rank == p:
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

    print(f"Rank {rank} Avg. for Part 1: {sum(part1):.3f} \n")
    print(f"Rank {rank} Avg. for Part 2: {sum(part2):.3f} \n")
    print(f"Rank {rank} Avg. for Part 3: {sum(part3):.3f} \n")
    print(f"Rank {rank} Avg. for Part 2_1: {sum(part2_1):.3f} \n")
    print(f"Rank {rank} Avg. for Part 2_2: {sum(part2_2):.3f} \n")
    print("****")

    return walks_summary
