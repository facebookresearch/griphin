from collections import OrderedDict, defaultdict
from contextlib import nullcontext
import time

import torch
from torch.distributed import rpc

from graph import GraphShard, VERTEX_ID_TYPE


def random_walk(shard_rrefs, num_machines, num_roots, walk_length, profile, profile_prefix):
    """Kernel Function of Distributed Random Walk

    :param shard_rrefs: Reference list of remote graph shards.
    :param num_machines: World size.
    :param num_roots: Number of root nodes in current machine.
    :param walk_length: Length of random walks.
    :return: walk_summary
    """

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=walk_length, skip_first=0, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_prefix)
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
        # init index "root_perm" that maps "u" to root nodes
        u = root_nodes
        root_perm = torch.arange(num_roots)

        # init shard dict for current step "shard_dict_u" and next step "shard_dict_v"
        shard_dict_u = OrderedDict({rank: torch.arange(num_roots)})
        shard_dict_v = defaultdict(list)

        if rank == 0:
            print(f"Starting Distributed Random Walk:"
                f" world_size={num_machines}, num_roots={num_roots}, walk_length={walk_length}")

        part1 = []
        part2 = []
        part3 = []
        part4 = []

        part2_1 = []
        part2_2 = []
        part2_3 = []

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
            tik_ = time.time()

            shard_dict_v.clear()

            # issue async RPC call
            futs = {}
            for p, index_u in shard_dict_u.items():
                if rank == p:
                    continue
                futs[p] = shard_rrefs[p].rpc_async().walk_one_step(u[index_u])

            tik_ = time.time()

            # overlap remote step with local step
            nid_local, shard_dict_local = local_shard.walk_one_step(u[shard_dict_u[rank]])

            tok_ = time.time()

            part4.append(tok_ - tik_)

            tok_ = time.time()

            part1.append(tok_-tik_)

            tik_ = time.time()
            # collect results and constructs shard_dict_v for next step
            nids_global, nids, offset = [], [], 0
            
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for p in shard_dict_u.keys():
                tik1_ = time.time()

                if rank == p:
                    nid, shard_dict = nid_local, shard_dict_local
                else:
                    nid, shard_dict = futs[p].wait()

                tok1_ = time.time()

                sum1 += tok1_ - tik1_

                nid_global = torch.empty_like(nid)
                

                sum2_ = 0
                for q, index_v in shard_dict.items():
                    tik2_s = time.time()
                    shard_dict_v[q].append(index_v + offset)  # merge shard dicts
                    tok2_s = time.time()
                    sum2_ += tok2_s - tik2_s 

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

            part2.append(tok_-tik_)


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

            part3.append(tok_-tik_)

            if profile:
                profiler.step()
        
        print("Rank: ", rank, " Avg. for Part 1:", sum(part1), "\n")
        print("Rank: ", rank, " Avg. for Part 2:", sum(part2), "\n")
        print("Rank: ", rank, " Avg. for Part 3:", sum(part3), "\n")
        print("Rank: ", rank, " Avg. for Part 4:", sum(part4), "\n")

        print("Rank: ", rank, " Avg. for Part 2_1:", sum(part2_1), "\n")
        print("Rank: ", rank, " Avg. for Part 2_2:", sum(part2_2), "\n")
        print("Rank: ", rank, " Avg. for Part 2_3:", sum(part2_3), "\n")

        print("****")

    return walks_summary