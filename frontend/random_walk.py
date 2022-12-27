from collections import OrderedDict, defaultdict

import torch
from torch.distributed import rpc

from frontend.graph import GraphShard, VERTEX_ID_TYPE


def random_walk(walker_rrefs, num_machines, num_roots, walk_length):
    rank = rpc.get_worker_info().id
    walker: GraphShard = walker_rrefs[rank].to_here()

    # root_nodes = torch.arange(num_roots, dtype=VERTEX_ID_TYPE)  # for test
    root_nodes = torch.randperm(walker.num_core_nodes, dtype=VERTEX_ID_TYPE)[:num_roots]
    root_perm = torch.arange(num_roots)  # index mapping the current step nodes to root nodes

    walks_summary = torch.full((num_roots, walk_length + 1), -1, dtype=root_nodes.dtype)
    walks_summary[:, 0] = walker.to_global(root_nodes)

    u = root_nodes  # current step nodes
    shard_dict_u = OrderedDict({rank: torch.arange(num_roots)})
    shard_dict_v = defaultdict(list)

    if rank == 0:
        print(f"Starting Distributed Random Walk:"
              f" world_size={num_machines}, num_roots={num_roots}, walk_length={walk_length}")

    for i in range(walk_length):
        shard_dict_v.clear()

        # remote call
        futs = {}
        for p, index_u in shard_dict_u.items():
            if rank == p:
                continue
            futs[p] = walker_rrefs[p].rpc_async().step(u[index_u])

        # Overlap local sampling with remote sampling:
        ret_local, shard_dict_local = walker.step(u[shard_dict_u[rank]])

        # collect results
        rets_global, rets = [], []
        offset = 0
        for p in shard_dict_u.keys():
            if rank == p:
                ret, shard_dict = ret_local, shard_dict_local
            else:
                ret, shard_dict = futs[p].wait()

            ret_global = torch.empty_like(ret)
            for q, index_v in shard_dict.items():
                shard_dict_v[q].append(index_v + offset)
                ret_global[index_v] = walker.to_global(ret[index_v], q)

            rets.append(ret)
            rets_global.append(ret_global)
            offset += ret.size(0)

        # update summary
        perm = torch.cat(list(shard_dict_u.values()))
        root_perm = root_perm.index_select(0, perm)
        walks_summary[root_perm, i + 1] = torch.cat(rets_global)

        # update node_index & shard_dict
        u = torch.cat(rets)  # shard_dict_u must have the same shard order with rets/rets_global
        shard_dict_u.clear()
        for q, index_v_list in sorted(shard_dict_v.items()):  # get sorted shard_dict_u
            shard_dict_u[q] = torch.cat(index_v_list)

    return walks_summary
