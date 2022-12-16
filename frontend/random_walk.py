from collections import OrderedDict

import torch
from torch.distributed import rpc

from frontend.graph import GraphShard


def random_walk(walker_rrefs, num_machines, num_roots, walk_length):
    rank = rpc.get_worker_info().id
    walker: GraphShard = walker_rrefs[rank].to_here()

    batch_size = walker.cluster_size
    root_nodes = torch.randperm(batch_size)[:num_roots]

    walks_summary = torch.full((num_roots, walk_length + 1), -1)
    walks_summary[:, 0] = walker.to_global(root_nodes)

    u = root_nodes  # current layer nodes u
    shard_dict_u = {rank: torch.arange(batch_size, dtype=root_nodes.dtype)}  # OrderedDict{shard_id: index}

    if rank == 0:
        print(f"Starting Distributed Random Walk:"
              f" world_size={num_machines}, num_roots={num_roots}, walk_length={walk_length}")

    for i in range(walk_length):
        futs, shard_dict_v = [], OrderedDict([(k, []) for k in range(num_machines)])

        # remote call
        for j, index in shard_dict_u.items():
            if rank == j:
                continue
            fut = walker_rrefs[j].rpc_async().step(u[index])
            futs.append(fut)

        # Overlap local sampling with remote sampling:
        ret_local, shard_dict_local = walker.step(u[shard_dict_u[rank]])

        # collect results
        rets_global = []  # for walks_summary only
        rets, ret, shard_dict = [], None, None
        offset, k = 0, 0
        for k in shard_dict_u.keys():
            if rank == k:
                ret = ret_local
                shard_dict = shard_dict_local
            else:
                ret, shard_dict = futs[k-1 if rank > k else k].wait()

            rets.append(ret)
            rets_global.append(walker.to_global(ret, k))
            for q, s_index in shard_dict.items():
                shard_dict_v[q].append(s_index + offset)

            offset += ret.size(0)

        # update results
        perm = torch.cat(list(shard_dict_u.values()))
        u[perm] = torch.cat(rets)
        walks_summary[perm, i+1] = torch.cat(rets_global)

        # update shard dict
        for q in shard_dict_v.keys():
            shard_dict_u[q] = torch.cat(shard_dict_v[q])

    return walks_summary
