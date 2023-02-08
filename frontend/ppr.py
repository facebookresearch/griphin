import torch
from torch.distributed import rpc

from graph import GraphShard, SSPPR, VERTEX_ID_TYPE


def forward_push_single(rrefs, num_source, alpha, epsilon):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = rrefs[rank].to_here()
    cluster_ptr = local_shard.cluster_ptr

    source_ids = torch.randperm(local_shard.num_core_nodes)[:num_source]
    results = []
    for epoch, target_id in enumerate(source_ids):
        ppr_model = SSPPR(cluster_ptr[-1], target_id, rank, cluster_ptr, alpha, epsilon)

        iteration = 0

        while True:
            v_ids, v_shard_ids = ppr_model.pop_activated_nodes()

            iteration += 1
            print('iter:', iteration, ', activated nodes:', len(v_ids))

            if len(v_ids) == 0:
                break
            for v_id, v_shard_id in zip(v_ids.tolist(), v_shard_ids.tolist()):
                v_id_ = torch.tensor([v_id], dtype=VERTEX_ID_TYPE)
                if v_shard_id == rank:
                    neighbor_infos = local_shard.batch_fetch_neighbor_infos(v_id_)
                else:
                    neighbor_infos = rrefs[v_shard_id].rpc_sync().batch_fetch_neighbor_infos(v_id_)
                # print(neighbor_infos)
                ppr_model.push(neighbor_infos, v_id_, torch.tensor([v_shard_id]))

        results.append(ppr_model.p)

    return results


def forward_push_batch(rrefs, alpha, epsilon):
    print('batch')
