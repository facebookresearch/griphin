import time

import torch
from torch.distributed import rpc

from graph import GraphShard, SSPPR, PPR, VERTEX_ID_TYPE


def cpp_push_single(rrefs, num_source, alpha, epsilon, log=False):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = rrefs[rank].to_here()

    time_pop = 0
    time_fetch_neighbor_local = 0
    time_fetch_neighbor_remote = 0
    time_push = 0

    source_ids = torch.randperm(local_shard.num_core_nodes)[:num_source]
    results = []
    for epoch, target_id in enumerate(source_ids):
        ppr_model = PPR(target_id, rank, alpha, epsilon)

        iteration = 0
        if log and rank == 0:
            print('\nSource Node:', epoch)

        while True:
            tik = time.time()
            v_ids, v_shard_ids = ppr_model.pop_activated_nodes()
            time_pop += time.time() - tik

            iteration += 1
            if log and rank == 0:
                print('iter:', iteration, ', activated nodes:', len(v_ids))

            if len(v_ids) == 0:
                break

            for v_id, v_shard_id in zip(v_ids.tolist(), v_shard_ids.tolist()):
                v_id_ = torch.tensor([v_id], dtype=VERTEX_ID_TYPE)

                tik = time.time()
                if v_shard_id == rank:
                    neighbor_infos = local_shard.batch_fetch_neighbor_infos(v_id_)
                    time_fetch_neighbor_local += time.time() - tik
                else:
                    neighbor_infos = rrefs[v_shard_id].rpc_sync().batch_fetch_neighbor_infos(v_id_)
                    time_fetch_neighbor_remote += time.time() - tik

                tik = time.time()
                ppr_model.push(neighbor_infos, v_id_, torch.tensor([v_shard_id]))
                time_push += time.time() - tik

        results.append(ppr_model.get_p()[2])

    if rank == 0:
        print(f'Time pop: {time_pop:.3f}s, '
              f'Time fetch local: {time_fetch_neighbor_local:.3f}s, '
              f'Time fetch remote: {time_fetch_neighbor_remote:.3f}s, '
              f'Time push: {time_push:.3f}s')

    return results


def cpp_push_batch(rrefs, num_source, alpha, epsilon, log=False):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = rrefs[rank].to_here()
    num_machines = len(rrefs)

    time_fetch_neighbor_local = 0
    time_fetch_neighbor_remote = 0
    time_push = 0

    results = []
    source_ids = torch.randperm(local_shard.num_core_nodes)[:num_source]
    for epoch, target_id in enumerate(source_ids):
        ppr_model = PPR(target_id, rank, alpha, epsilon)

        iteration = 0
        if log and rank == 0:
            print('\nSource Node:', epoch)

        while True:
            v_ids, v_shard_ids = ppr_model.pop_activated_nodes()

            iteration += 1
            if log and rank == 0:
                print('iter:', iteration, ', activated nodes:', len(v_ids))

            if len(v_ids) == 0:
                break

            v_ids_dict, v_shard_ids_dict = {}, {}
            for j in range(num_machines):
                mask = v_shard_ids == j
                v_ids_dict[j], v_shard_ids_dict[j] = v_ids[mask], v_shard_ids[mask]

            futs = {}
            for j, j_v_ids in v_ids_dict.items():
                if rank == j or len(j_v_ids) == 0:
                    continue
                futs[j] = rrefs[j].rpc_async().batch_fetch_neighbor_infos(j_v_ids)

            tik = time.time()
            local_neighbor_infos = local_shard.batch_fetch_neighbor_infos(v_ids_dict[rank])
            time_fetch_neighbor_local += time.time() - tik

            tik = time.time()
            remote_infos, remote_v_ids, remote_shard_ids = [], [], []
            for j, fut in futs.items():
                infos = fut.wait()
                remote_infos += infos
                remote_v_ids.append(v_ids_dict[j])
                remote_shard_ids.append(v_shard_ids_dict[j])
            time_fetch_neighbor_remote += time.time() - tik

            tik = time.time()
            # push to neighborhood from local shard
            ppr_model.push(local_neighbor_infos, v_ids_dict[rank], v_shard_ids_dict[rank])
            # push to neighborhood from remote shard
            if len(remote_infos) > 0:
                ppr_model.push(remote_infos, torch.cat(remote_v_ids), torch.cat(remote_shard_ids))
            time_push += time.time() - tik

        results.append(ppr_model.get_p()[2])

    if rank == 0:
        print(f'Time fetch local: {time_fetch_neighbor_local:.3f}s, '
              f'Time fetch remote: {time_fetch_neighbor_remote:.3f}s, '
              f'Time push: {time_push:.3f}s')

    return results


def python_push_single(rrefs, num_source, alpha, epsilon, log=False):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = rrefs[rank].to_here()

    time_fetch_neighbor_local = 0
    time_fetch_neighbor_remote = 0
    time_push = 0

    results = []
    source_ids = torch.randperm(local_shard.num_core_nodes)[:num_source]
    for epoch, target_id in enumerate(source_ids):
        ppr_model = SSPPR(target_id, rank, alpha, epsilon)

        iteration = 0
        if log and rank == 0:
            print('\nSource Node:', epoch)

        while True:
            v_ids, v_shard_ids = ppr_model.pop_activated_nodes()

            iteration += 1
            if log and rank == 0:
                print('iter:', iteration, ', activated nodes:', len(v_ids))

            if len(v_ids) == 0:
                break

            for v_id, v_shard_id in zip(v_ids.tolist(), v_shard_ids.tolist()):
                v_id_ = torch.tensor([v_id], dtype=VERTEX_ID_TYPE)

                tik = time.time()
                if v_shard_id == rank:
                    neighbor_infos = local_shard.batch_fetch_neighbor_infos(v_id_)
                    time_fetch_neighbor_local += time.time() - tik
                else:
                    neighbor_infos = rrefs[v_shard_id].rpc_sync().batch_fetch_neighbor_infos(v_id_)
                    time_fetch_neighbor_remote += time.time() - tik

                tik = time.time()
                ppr_model.push(neighbor_infos, v_id_, torch.tensor([v_shard_id]))
                time_push += time.time() - tik

        results.append(ppr_model.p)

    if rank == 0:
        print(f'Time fetch local: {time_fetch_neighbor_local:.3f}s, '
              f'Time fetch remote: {time_fetch_neighbor_remote:.3f}s, '
              f'Time push: {time_push:.3f}s')

    return results


def python_push_batch(rrefs, num_source, alpha, epsilon, log=False):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = rrefs[rank].to_here()
    num_machines = len(rrefs)

    time_fetch_neighbor_local = 0
    time_fetch_neighbor_remote = 0
    time_push = 0

    results = []
    source_ids = torch.randperm(local_shard.num_core_nodes)[:num_source]
    for epoch, target_id in enumerate(source_ids):
        ppr_model = SSPPR(target_id, rank, alpha, epsilon)

        iteration = 0
        if log and rank == 0:
            print('\nSource Node:', epoch)

        while True:
            v_ids, v_shard_ids = ppr_model.pop_activated_nodes()

            iteration += 1
            if log and rank == 0:
                print('iter:', iteration, ', activated nodes:', len(v_ids))

            if len(v_ids) == 0:
                break

            v_ids_dict, v_shard_ids_dict = {}, {}
            for j in range(num_machines):
                mask = v_shard_ids == j
                v_ids_dict[j], v_shard_ids_dict[j] = v_ids[mask], v_shard_ids[mask]

            futs = {}
            for j, j_v_ids in v_ids_dict.items():
                if rank == j or len(j_v_ids) == 0:
                    continue
                futs[j] = rrefs[j].rpc_async().batch_fetch_neighbor_infos(j_v_ids)

            tik = time.time()
            local_neighbor_infos = local_shard.batch_fetch_neighbor_infos(v_ids_dict[rank])
            time_fetch_neighbor_local += time.time() - tik

            tik = time.time()
            remote_infos, remote_v_ids, remote_shard_ids = [], [], []
            for j, fut in futs.items():
                infos = fut.wait()
                remote_infos += infos
                remote_v_ids.append(v_ids_dict[j])
                remote_shard_ids.append(v_shard_ids_dict[j])
            time_fetch_neighbor_remote += time.time() - tik

            tik = time.time()
            # push to neighborhood from local shard
            ppr_model.push(local_neighbor_infos, v_ids_dict[rank], v_shard_ids_dict[rank])
            # push to neighborhood from remote shard
            if len(remote_infos) > 0:
                ppr_model.push(remote_infos, torch.cat(remote_v_ids), torch.cat(remote_shard_ids))
            time_push += time.time() - tik

        results.append(ppr_model.p)

    if rank == 0:
        print(f'Time fetch local: {time_fetch_neighbor_local:.3f}s, '
              f'Time fetch remote: {time_fetch_neighbor_remote:.3f}s, '
              f'Time push: {time_push:.3f}s')

    return results
