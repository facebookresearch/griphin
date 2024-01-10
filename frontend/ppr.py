#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import time

import torch
from torch.distributed import rpc

from graph import SSPPR, PPR, VERTEX_ID_TYPE, DistGraphStorage, SHARD_ID_TYPE
import graph_engine


def forward_push(machine_rank, process_rank, rrefs, num_machines, num_threads,
                 source_ids, alpha, epsilon, log=False, output_name=''):
    """
        Overlapped version of cpp_batch
    """
    dist_gs = DistGraphStorage(rrefs, machine_rank)

    time_pop = 0
    time_local_fetch = 0
    time_local_push = 0
    time_remote_push = 0
    time_total = 0

    num_source = source_ids.shape[0]
    if len(output_name) > 0:
        """
        TODO: There is a bug in Torch RPC (2.0.1) that causes a worker to be destroyed
        if it has been idle for a period of time. To bypass this issue, I manually
        introduced more tasks (source nodes) to ensure that all workers finish their 
        originally assigned tasks before they are terminated.
        """
        source_ids = torch.cat([source_ids, source_ids[:int(num_source * 0.5)]])

    results = []
    for epoch, target_id in enumerate(source_ids):
        ppr_model = graph_engine.PPR(target_id, machine_rank, alpha, epsilon, num_threads)

        iteration = 0
        if log and machine_rank == 0 and process_rank == 0:
            print('\nSource Node:', target_id.item(), 'Worker: ', rpc.get_worker_info().id)

        while True:
            tik = time.time()
            v_ids, v_shard_ids = ppr_model.pop_activated_nodes()
            time_pop += time.time() - tik

            iteration += 1
            # if log and machine_rank == 1 and process_rank == 0:
            #     print('iter:', iteration, ', num activated nodes:', len(v_ids))

            if len(v_ids) == 0:
                break

            # compute shard masks
            v_ids_dict, v_shard_ids_dict = {}, {}
            for j in range(num_machines):
                mask = v_shard_ids == j
                v_ids_dict[j], v_shard_ids_dict[j] = v_ids[mask], v_shard_ids[mask]

            tik_ = time.time()

            futs = {}
            for j, j_v_ids in v_ids_dict.items():
                if j == machine_rank or len(j_v_ids) == 0:
                    continue
                # futs[j] = rrefs[j].rpc_async().get_neighbor_infos_remote(j_v_ids)
                futs[j] = dist_gs.get_neighbor_infos(j, j_v_ids)

            tik = time.time()
            # local_neighbor_infos = local_shard.get_neighbor_infos_local(v_ids_dict[machine_rank])
            local_neighbor_infos = dist_gs.get_neighbor_infos(machine_rank, v_ids_dict[machine_rank])
            time_local_fetch += time.time() - tik

            tik = time.time()
            ppr_model.push(local_neighbor_infos, v_ids_dict[machine_rank], v_shard_ids_dict[machine_rank])
            time_local_push += time.time() - tik

            for j, fut in futs.items():
                # print('Reading current req from', j, v_ids_dict[j])
                infos = fut.wait()
                tik = time.time()
                ppr_model.push(infos, v_ids_dict[j], v_shard_ids_dict[j])
                time_remote_push += time.time() - tik

            time_total += time.time() - tik_

        res = ppr_model.get_p()
        results.append(res)

        if len(results) == num_source:
            print(f'Machine {machine_rank} Process {process_rank} Finished, Num source: {num_source}')
            if len(output_name) > 0:
                # default output_name: 'intermediate/ogbn-products_{}_{}.pt'
                torch.save(results, output_name.format(machine_rank, process_rank))

    time_dict = {
        # 'Time pop': time_pop,
        'Time fetch local': time_local_fetch,
        'Time push local': time_local_push,
        'Time push remote': time_remote_push,
        'Time total': time_total
    }

    if machine_rank == process_rank == 0 and log:
        print(f'Machine rank: {machine_rank}, Process rank: {process_rank}')
        for k, v in time_dict.items():
            print(f'{k}: {v:.3f}s')

    return results, time_dict


def cpp_push_single(machine_rank, process_rank, rrefs, num_machines, num_threads,
                    source_ids, alpha, epsilon, log=False):
    time_start = time.time()

    rank = rpc.get_worker_info().id
    local_shard = rrefs[rank].to_here()

    time_pop = 0
    time_fetch_neighbor_local = 0
    time_fetch_neighbor_remote = 0
    time_push = 0

    results = []
    for epoch, target_id in enumerate(source_ids):
        ppr_model = graph_engine.PPR(target_id, machine_rank, alpha, epsilon, num_threads)

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
                ppr_model.push(neighbor_infos, v_id_, torch.tensor([v_shard_id], dtype=SHARD_ID_TYPE))
                time_push += time.time() - tik

        res = ppr_model.get_p()
        results.append(res)

    time_total = time.time() - time_start

    time_dict = {
        'Time pop': time_pop,
        'Time fetch local': time_fetch_neighbor_local,
        'Time fetch remote': time_fetch_neighbor_remote,
        'Time push': time_push,
        'Time total': time_total
    }

    if machine_rank == process_rank == 0:
        print(f'Time pop: {time_pop:.3f}s, '
              f'Time fetch local: {time_fetch_neighbor_local:.3f}s, '
              f'Time fetch remote: {time_fetch_neighbor_remote:.3f}s, '
              f'Time push: {time_push:.3f}s, '
              f'Time total: {time_total:.3f}s')

    return results, time_dict


def cpp_push_batch(machine_rank, process_rank, rrefs, num_machines, num_threads,
                   source_ids, alpha, epsilon, log=False, **kwargs):
    time_start = time.time()

    dist_gs = DistGraphStorage(rrefs, machine_rank)

    time_pop = 0
    time_fetch_local = 0
    time_fetch_remote = 0
    time_push_remote = 0
    time_push_local = 0
    num_local_fetch = 0
    num_total_fetch = 0

    results = []
    for epoch, target_id in enumerate(source_ids):
        ppr_model = graph_engine.PPR(target_id, machine_rank, alpha, epsilon, num_threads)

        iteration = 0
        if log and machine_rank == process_rank == 0:
            print('\nSource Node:', target_id)

        while True:
            tik = time.time()
            v_ids, v_shard_ids = ppr_model.pop_activated_nodes()
            time_pop += time.time() - tik
            num_total_fetch += v_ids.size(-1)

            iteration += 1
            if log and machine_rank == process_rank == 0:
                print('iter:', iteration, ', num activated nodes:', len(v_ids))

            if len(v_ids) == 0:
                break

            v_ids_dict, v_shard_ids_dict = {}, {}
            for j in range(num_machines):
                mask = v_shard_ids == j
                v_ids_dict[j], v_shard_ids_dict[j] = v_ids[mask], v_shard_ids[mask]

            # fetch neighborhood from local shard
            tik = time.time()
            # local_neighbor_infos = rrefs[rank].to_here().get_neighbor_infos_local(v_ids_dict[machine_rank])
            local_neighbor_infos = dist_gs.get_neighbor_infos(machine_rank, v_ids_dict[machine_rank])
            time_fetch_local += time.time() - tik
            num_local_fetch += v_ids_dict[machine_rank].size(-1)

            # fetch neighborhood from remote shard
            tik_ = time.time()
            futs = {}
            for j, j_v_ids in v_ids_dict.items():
                if j == machine_rank or len(j_v_ids) == 0:
                    continue
                # futs[j] = rrefs[j].rpc_async().get_neighbor_infos_remote(j_v_ids)
                futs[j] = dist_gs.get_neighbor_infos(j, j_v_ids)
            infos = {}
            for j, fut in futs.items():
                infos[j] = fut.wait()
            time_fetch_remote += time.time() - tik_

            # push to neighborhood from remote shard
            tik = time.time()
            for j, info in infos.items():
                ppr_model.push(info, v_ids_dict[j], v_shard_ids_dict[j])
            time_push_remote += time.time() - tik

            # push to neighborhood from local shard
            tik = time.time()
            ppr_model.push(local_neighbor_infos, v_ids_dict[machine_rank], v_shard_ids_dict[machine_rank])
            time_push_local += time.time() - tik

        res = ppr_model.get_p()
        results.append(res)

    time_total = time.time() - time_start

    time_dict = {
        # 'Time pop': time_pop,
        'Time fetch local': time_fetch_local,
        'Time fetch remote': time_fetch_remote,
        'Time push remote': time_push_remote,
        'Time push local': time_push_local,
        'Time total': time_total,
        # 'Ratio fetch remote': 1 - num_local_fetch/num_total_fetch,
    }

    if machine_rank == process_rank == 0:
        print(f'Machine rank: {machine_rank}, Process rank: {process_rank}')
        for k, v in time_dict.items():
            print(f'{k}: {v:.3f}s')

    return results, time_dict


def python_push_single(machine_rank, process_rank, rrefs, num_machines, num_threads,
                       source_ids, alpha, epsilon, log=False):
    rank = rpc.get_worker_info().id
    local_shard = rrefs[rank].to_here()

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
    local_shard = rrefs[rank].to_here()

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
