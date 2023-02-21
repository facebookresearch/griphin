import time

import torch
from torch.distributed import rpc

from graph import GraphShard, SSPPR, PPR, VERTEX_ID_TYPE


def cpp_push_single(machine_rank, process_rank, rrefs, num_machines, num_threads,
                    source_ids, alpha, epsilon, log=False):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = rrefs[rank].to_here()

    time_pop = 0
    time_fetch_neighbor_local = 0
    time_fetch_neighbor_remote = 0
    time_push = 0

    results = []
    for epoch, target_id in enumerate(source_ids):
        ppr_model = PPR(target_id, machine_rank, alpha, epsilon)

        iteration = 0
        if log and machine_rank == process_rank == 0:
            print('\nSource Node:', epoch)

        while True:
            tik = time.time()
            v_ids, v_shard_ids = ppr_model.pop_activated_nodes()
            time_pop += time.time() - tik

            iteration += 1
            if log and machine_rank == process_rank == 0:
                print('iter:', iteration, ', activated nodes:', len(v_ids))

            if len(v_ids) == 0:
                break

            for v_id, v_shard_id in zip(v_ids.tolist(), v_shard_ids.tolist()):
                v_id_ = torch.tensor([v_id], dtype=VERTEX_ID_TYPE)

                tik = time.time()
                if v_shard_id == machine_rank:
                    neighbor_infos = local_shard.get_neighbor_infos(v_id_)
                    time_fetch_neighbor_local += time.time() - tik
                else:
                    neighbor_infos = rrefs[v_shard_id].rpc_sync().get_neighbor_infos(v_id_)
                    time_fetch_neighbor_remote += time.time() - tik

                tik = time.time()
                ppr_model.push(neighbor_infos, v_id_, torch.tensor([v_shard_id]), num_threads)
                time_push += time.time() - tik

        res = ppr_model.get_p()
        results.append(res[2])

    if machine_rank == process_rank == 0:
        print(f'Time pop: {time_pop:.3f}s, '
              f'Time fetch local: {time_fetch_neighbor_local:.3f}s, '
              f'Time fetch remote: {time_fetch_neighbor_remote:.3f}s, '
              f'Time push: {time_push:.3f}s')

    return results, time_pop, time_fetch_neighbor_local, time_fetch_neighbor_remote, time_push


def cpp_push_batch(machine_rank, process_rank, rrefs, num_machines, num_threads,
                   source_ids, alpha, epsilon, log=False):
    rank = rpc.get_worker_info().id
    local_shard = rrefs[rank].to_here()  # only available for local shard

    time_pop = 0
    time_fetch_neighbor_local = 0
    time_fetch_neighbor_remote = 0
    time_push = 0

    results = []
    for epoch, target_id in enumerate(source_ids):
        ppr_model = PPR(target_id, machine_rank, alpha, epsilon)

        iteration = 0
        if log and machine_rank == process_rank == 0:
            print('\nSource Node:', target_id)

        while True:
            tik = time.time()
            v_ids, v_shard_ids = ppr_model.pop_activated_nodes()
            time_pop += time.time() - tik

            iteration += 1
            if log and machine_rank == process_rank == 0:
                print('iter:', iteration, ', num activated nodes:', len(v_ids))

            if len(v_ids) == 0:
                break

            v_ids_dict, v_shard_ids_dict = {}, {}
            for j in range(num_machines):
                mask = v_shard_ids == j
                v_ids_dict[j], v_shard_ids_dict[j] = v_ids[mask], v_shard_ids[mask]

            tik = time.time()
            local_neighbor_infos = local_shard.get_neighbor_infos(v_ids_dict[machine_rank])
            time_fetch_neighbor_local += time.time() - tik

            tik_ = time.time()
            futs = {}
            for j, j_v_ids in v_ids_dict.items():
                if j == machine_rank or len(j_v_ids) == 0:
                    continue
                futs[j] = rrefs[j].rpc_async().get_neighbor_infos(j_v_ids)
            remote_infos, remote_v_ids, remote_shard_ids = [], [], []
            for j, fut in futs.items():
                infos = fut.wait()
                remote_infos += infos
                remote_v_ids.append(v_ids_dict[j])
                remote_shard_ids.append(v_shard_ids_dict[j])
            time_fetch_neighbor_remote += time.time() - tik_

            tik = time.time()
            # push to neighborhood from local shard
            ppr_model.push(local_neighbor_infos, v_ids_dict[machine_rank], v_shard_ids_dict[machine_rank], num_threads)
            # push to neighborhood from remote shard
            if len(remote_infos) > 0:
                ppr_model.push(remote_infos, torch.cat(remote_v_ids), torch.cat(remote_shard_ids), num_threads)
            time_push += time.time() - tik

        res = ppr_model.get_p()
        results.append(res[2])

    if machine_rank == process_rank == 0:
        print(f'Time pop: {time_pop:.3f}s, '
              f'Time fetch local: {time_fetch_neighbor_local:.3f}s, '
              f'Time fetch remote: {time_fetch_neighbor_remote:.3f}s, '
              f'Time push: {time_push:.3f}s')

    return results, time_pop, time_fetch_neighbor_local, time_fetch_neighbor_remote, time_push


def python_push_single(machine_rank, process_rank, rrefs, num_machines, num_threads,
                        source_ids, alpha, epsilon, log=False):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = rrefs[rank].to_here()

    time_pop = 0
    time_fetch_neighbor_local = 0
    time_fetch_neighbor_remote = 0
    time_push = 0

    results = []
    for epoch, target_id in enumerate(source_ids):
        ppr_model = SSPPR(target_id, machine_rank, alpha, epsilon)

        iteration = 0
        if log and machine_rank == process_rank == 0:
            print('\nSource Node:', epoch)

        while True:
            tik = time.time()
            v_ids, v_shard_ids = ppr_model.pop_activated_nodes()
            time_pop += time.time() - tik

            iteration += 1
            if log and machine_rank == process_rank == 0:
                print('iter:', iteration, ', activated nodes:', len(v_ids))

            if len(v_ids) == 0:
                break

            for v_id, v_shard_id in zip(v_ids.tolist(), v_shard_ids.tolist()):
                v_id_ = torch.tensor([v_id], dtype=VERTEX_ID_TYPE)

                tik = time.time()
                if v_shard_id == machine_rank:
                    neighbor_infos = local_shard.get_neighbor_infos(v_id_)
                    time_fetch_neighbor_local += time.time() - tik
                else:
                    neighbor_infos = rrefs[v_shard_id].rpc_sync().get_neighbor_infos(v_id_)
                    time_fetch_neighbor_remote += time.time() - tik

                tik = time.time()
                ppr_model.push(neighbor_infos, v_id_, torch.tensor([v_shard_id]))
                time_push += time.time() - tik

        results.append(ppr_model.p)

    if machine_rank == process_rank == 0:
        print(f'Time pop: {time_pop:.3f}s, '
              f'Time fetch local: {time_fetch_neighbor_local:.3f}s, '
              f'Time fetch remote: {time_fetch_neighbor_remote:.3f}s, '
              f'Time push: {time_push:.3f}s')

    return results, time_pop, time_fetch_neighbor_local, time_fetch_neighbor_remote, time_push


def python_push_batch(machine_rank, process_rank, rrefs, num_machines, num_threads,
                      source_ids, alpha, epsilon, log=False):
    rank = rpc.get_worker_info().id
    local_shard: GraphShard = rrefs[rank].to_here()

    time_pop = 0
    time_fetch_neighbor_local = 0
    time_fetch_neighbor_remote = 0
    time_push = 0

    results = []
    for epoch, target_id in enumerate(source_ids):
        ppr_model = SSPPR(target_id, machine_rank, alpha, epsilon)

        iteration = 0
        if log and machine_rank == process_rank == 0:
            print('\nSource Node:', epoch)

        while True:
            tik = time.time()
            v_ids, v_shard_ids = ppr_model.pop_activated_nodes()
            time_pop += time.time() - tik

            iteration += 1
            if log and machine_rank == process_rank == 0:
                print('iter:', iteration, ', activated nodes:', len(v_ids))

            if len(v_ids) == 0:
                break

            v_ids_dict, v_shard_ids_dict = {}, {}
            for j in range(num_machines):
                mask = v_shard_ids == j
                v_ids_dict[j], v_shard_ids_dict[j] = v_ids[mask], v_shard_ids[mask]

            tik = time.time()
            local_neighbor_infos = local_shard.get_neighbor_infos(v_ids_dict[machine_rank])
            time_fetch_neighbor_local += time.time() - tik

            tik_ = time.time()
            futs = {}
            for j, j_v_ids in v_ids_dict.items():
                if machine_rank == j or len(j_v_ids) == 0:
                    continue
                futs[j] = rrefs[j].rpc_async().get_neighbor_infos(j_v_ids)
            remote_infos, remote_v_ids, remote_shard_ids = [], [], []
            for j, fut in futs.items():
                infos = fut.wait()
                remote_infos += infos
                remote_v_ids.append(v_ids_dict[j])
                remote_shard_ids.append(v_shard_ids_dict[j])
            time_fetch_neighbor_remote += time.time() - tik_

            tik = time.time()
            # push to neighborhood from local shard
            ppr_model.push(local_neighbor_infos, v_ids_dict[machine_rank], v_shard_ids_dict[machine_rank])
            # push to neighborhood from remote shard
            if len(remote_infos) > 0:
                ppr_model.push(remote_infos, torch.cat(remote_v_ids), torch.cat(remote_shard_ids))
            time_push += time.time() - tik

        results.append(ppr_model.p)

    if machine_rank == process_rank == 0:
        print(f'Time pop: {time_pop:.3f}s, '
              f'Time fetch local: {time_fetch_neighbor_local:.3f}s, '
              f'Time fetch remote: {time_fetch_neighbor_remote:.3f}s, '
              f'Time push: {time_push:.3f}s')

    return results, time_pop, time_fetch_neighbor_local, time_fetch_neighbor_remote, time_push


# def cpp_push_batch(rrefs, num_source, alpha, epsilon, log=False):
#     rank = rpc.get_worker_info().id
#     local_shard: GraphShard = rrefs[rank].to_here()
#     source_nodes = torch.randperm(local_shard.num_core_nodes)[:num_source]
#
#     num_process = 1
#     num_data = int(num_source / num_process)
#
#     with mp.Pool(num_process) as pool:
#         futs = []
#         for i in range(num_process - 1):
#             start, end = i * num_data, (i+1) * num_data
#             futs.append(
#                 pool.apply_async(local_push, args=(
#                     source_nodes[start:end],
#                     rrefs, rank, alpha, epsilon, log
#                 ))
#             )
#         futs.append(
#             pool.apply_async(local_push, args=(
#                 source_nodes[(num_process - 1) * num_data:],
#                 rrefs, rank, alpha, epsilon, log
#             ))
#         )
#         print(source_nodes[(num_process - 1) * num_data:])
#         results = [fut.get() for fut in futs]
#
#     return results
