#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse
import os
from functools import partial
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from torch import Tensor
from torch.distributed import rpc
from torch.distributed.rpc import remote, RRef
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm import tqdm

from torch_geometric.nn import SAGEConv

from graph import GraphDataManager
from ppr import forward_push
from utils import get_data_path


parser = argparse.ArgumentParser()
parser.add_argument('--num_machines', type=int, default=4, help='number of machines (simulated as processes)')
parser.add_argument('--alpha', type=float, default=0.462, help='teleport probability')
parser.add_argument('--epsilon', type=float, default=1e-6, help='maximum residual')
parser.add_argument('--num_threads', type=int, default=1, help='number of threads used in push operation')
parser.add_argument('--worker_name', type=str, default='worker{}', help='name of workers, formatted by rank')
parser.add_argument('--file_path', type=str, default='', help='path to dataset')
parser.add_argument('--log', action='store_true', help='whether to log breakdown runtime')


class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


def extract_core_global_ids(parts):
    part_core_global_ids = []
    for i in range(len(parts)):
        part = parts[i]
        core_mask = part.ndata['inner_node'].type(torch.bool)
        part_global_id = part.ndata['orig_id']
        part_core_global_ids.append(part_global_id[core_mask])
    return part_core_global_ids


def convert_batch_data(ppr_res, batch_index, rank, dataset, K):
    num_parts = len(dataset['part_core_global_ids'])
    batch_size = batch_index.shape[0]

    def get_global_id(local_ids, shard_ids):
        local_ids = local_ids.to(torch.long)
        global_ids = torch.empty_like(local_ids)
        for j in range(num_parts):
            mask = shard_ids == j
            if mask.sum() == 0: continue
            global_ids[mask] = dataset['part_core_global_ids'][j][local_ids[mask]]
        return global_ids

    batch_local_ids = [batch_index]
    batch_shard_ids = [torch.full((batch_size,), rank)]
    for i in range(len(ppr_res)):
        val, idx = torch.sort(ppr_res[i][2], descending=True)
        top_k_index = idx[:K]
        batch_local_ids.append(ppr_res[i][0][top_k_index])
        batch_shard_ids.append(ppr_res[i][1][top_k_index])
    batch_local_ids, batch_shard_ids = torch.cat(batch_local_ids), torch.cat(batch_shard_ids)

    subset = get_global_id(batch_local_ids, batch_shard_ids)
    subset, inv = subset.unique(return_inverse=True)
    batch_node_global_id = subset[inv[:batch_size]]

    sub_edge_index = subgraph(subset, dataset['edge_index'])[0]
    sub_edge_index = sub_edge_index.unique(return_inverse=True)[1]

    return Data(dataset['X'][subset],
                sub_edge_index,
                y=dataset['y'][batch_node_global_id].view(-1),
                ego_idx=inv[:batch_size])


def train(rank, args, world_size, rrefs, dataset):
    # print(rank, dataset)
    # simulate distributed graph storage scenario
    convert_batch = partial(convert_batch_data, rank=rank, dataset=dataset, K=100)
    num_features, num_classes = dataset['X'].shape[-1], dataset['y'].max().item()+1

    # Init Dist Training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.manual_seed(12345)

    model = SAGE(num_features, 256, num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loader = DataLoader(torch.arange(500000), batch_size=32, shuffle=True)

    for epoch in range(1, 21):
        model.train()

        for batch_index in tqdm(loader):
            # compute ppr results according to batch index
            ppr_res, _ = forward_push(rank, 0, rrefs, args.num_machines, args.num_threads,
                                      batch_index, args.alpha, args.epsilon, args.log)
            # convert ppr results to PyG Data
            batch = convert_batch(ppr_res, batch_index)  # Access Entire Graph
            batch = batch.to(rank)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[batch.ego_idx]
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()

        dist.barrier()

        if rank == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        dist.barrier()

    dist.destroy_process_group()


def run(rank, args, world_size):
    # Init Dist Graph Engine
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    rpc.init_rpc(
        args.worker_name.format(rank),
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=4, rpc_timeout=240)
    )

    if rank == 0:
        data_manager_rrefs: List[RRef] = []
        for machine_rank in range(0, args.num_machines):
            info = rpc.get_worker_info(args.worker_name.format(machine_rank))
            data_manager_rrefs.append(
                remote(info, GraphDataManager, args=(
                    machine_rank, args.file_path, args.worker_name, args.num_machines, 0
                ))
            )

        rrefs = [None for _ in range(args.num_machines)]
        for machine_rank in range(0, args.num_machines):
            res = data_manager_rrefs[machine_rank].rpc_sync().get_graph_rrefs()
            rrefs[machine_rank] = res[0]

        print('Assembling graph shard references finished')

        # load entire graph data
        og, y = DglNodePropPredDataset(name='ogbn-products', root='/data/gangda/dgl')[0]
        dataset = dict(X=og.ndata['feat'], y=y)
        dataset['edge_index'] = torch.load(os.path.join(args.file_path, 'dgl_edge_index.pt'))
        parts = torch.load(os.path.join(args.file_path, 'metis_partitions.pt'))
        dataset['part_core_global_ids'] = extract_core_global_ids(parts)

        futs = []
        for machine_rank in range(0, args.num_machines):
            futs.append(
                rpc.rpc_async(
                    args.worker_name.format(machine_rank),
                    train,
                    args=(machine_rank, args, world_size, rrefs, dataset),
                )
            )
        for fut in futs:
            fut.wait()

    rpc.shutdown()


if __name__ == '__main__':
    args = parser.parse_args()
    if len(args.file_path) == 0:
        args.file_path = os.path.join(get_data_path(), 'ogbn-products-p{}'.format(args.num_machines))

    world_size = args.num_machines
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(args, world_size), nprocs=world_size, join=True)
