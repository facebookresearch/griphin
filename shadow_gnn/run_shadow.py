#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
import dgl
import dgl.sparse as dglsp
import sklearn
from ogb.nodeproppred import Evaluator, DglNodePropPredDataset, PygNodePropPredDataset
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric import seed_everything
from torch_geometric.data import Data, Batch
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import subgraph, dropout_edge, add_remaining_self_loops
from torch_sparse import SparseTensor
from torchmetrics import Accuracy
from tqdm import tqdm

from torch_geometric.nn import GATConv, GCNConv, CorrectAndSmooth

from model import GATConvWithNorm
from load_shm import fetch_datas_from_shm

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dgl_path', type=str, default='/data/gangda/dgl')
parser.add_argument('--data_path', type=str, default='data/ogbn-products-p2', help='graph shards dir')
parser.add_argument('--file_path', type=str, default='intermediate')
parser.add_argument('--ckpt_path', type=str, default='checkpoint')
parser.add_argument('--ckpt_file', type=str, default='', help='existing ckpt, skip training if not empty')
parser.add_argument('--out_file', type=str, default='',
                    help='existing out logits, skip training and testing if not empty')

parser.add_argument('--data_name', type=str, default='ogbn-products', choices=["ogbn-products"])
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=35)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--decay', type=float, default=0, help='weight decay')

parser.add_argument('--model', type=str, default='GAT-NORM-ACT')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.35)
parser.add_argument('--dropedge', type=float, default=0.1)
parser.add_argument('--pool', type=str, default='center', choices=["center", "max"])
parser.add_argument('--norm', action='store_true', help='enable layer norm')
parser.add_argument('--stand', action='store_true', help='enable feature standardization')
parser.add_argument('--feat_aug', action='store_true', help='augment feature with Label Prop')
parser.add_argument('--cs', action='store_true', help='correct & smooth')

# debugging
parser.add_argument('--ns', action='store_true', help='use neighbor sampling')


class ResPool(torch.nn.Module):
    def __init__(self, in_channels, out_channels, type_pool, dropout):
        super().__init__()
        self.type_pool = type_pool
        if type_pool == 'center':
            return

        self.transform = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(2 * in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(out_channels, eps=1e-9)
        )

    @staticmethod
    def _max_residue(feats):
        return torch.max(torch.stack(feats, dim=0), dim=0).values

    def forward(self, feats, ego_index, subg_offsets):
        if self.type_pool == 'center':
            return feats[-1][ego_index]
        elif self.type_pool == 'max':
            idx = torch.arange(feats[-1].shape[0], device=feats[-1].device)
            feat_pools, feat_roots = [], []
            for feat in feats:
                feat_pools.append(
                    F.embedding_bag(idx, feat, subg_offsets, mode=self.type_pool))
                feat_roots.append(feat[ego_index])
            feat_pool = self._max_residue(feat_pools)
            feat_root = self._max_residue(feat_roots)
            x = torch.cat([feat_root, feat_pool], dim=1)
        else:
            raise NotImplementedError

        return self.transform(x)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model='GAT', heads=4,
                 num_layers=5, dropout=0.35, dropedge=0.1, pooling='center', norm=False):
        super().__init__()
        self.dropout = dropout
        self.dropedge = dropedge
        self.pooling = pooling
        self.model = model.upper()

        self.convs = torch.nn.ModuleList()
        if self.model == 'GCN':
            self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
        elif self.model == 'GAT':
            assert hidden_channels % heads == 0
            single_head_channels = hidden_channels // heads
            self.convs.append(GATConv(in_channels, single_head_channels, heads))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_channels, single_head_channels, heads))
        elif self.model == 'GAT-NORM-ACT':
            assert hidden_channels % heads == 0
            single_head_channels = hidden_channels // heads
            self.convs.append(GATConvWithNorm(in_channels, single_head_channels, heads))
            for _ in range(num_layers - 1):
                self.convs.append(GATConvWithNorm(hidden_channels, single_head_channels, heads))
        else:
            raise NotImplementedError

        self.act = 'ACT' not in self.model

        self.norms = None
        if norm and 'NORM' not in self.model:
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers):
                # self.norms.append(torch.nn.BatchNorm1d(multihead_channels))
                self.norms.append(torch.nn.LayerNorm(hidden_channels, eps=1e-9))

        self.res_pool = ResPool(
            hidden_channels,
            hidden_channels,
            pooling,
            dropout
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, out_channels),
            torch.nn.LayerNorm(out_channels, eps=1e-9)
        )

    def forward(self, x: Tensor, edge_index: Tensor, ego_index: Tensor, subg_offsets=None) -> Tensor:
        edge_index = dropout_edge(edge_index, p=self.dropedge, training=self.training, force_undirected=False)[0]
        # edge_index = dropout_edge(edge_index, p=self.dropedge, training=self.training, force_undirected=True)[0]
        edge_weight = None

        if self.model == 'GCN':
            edge_index, edge_weight = gcn_norm(edge_index)

        xs = []
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight)
            if self.act:
                x = F.relu(x)
            if self.norms is not None:
                x = self.norms[i](x)
            xs.append(x)

        x = self.res_pool(xs, ego_index, subg_offsets)

        x = F.normalize(x, p=2, dim=1)

        x = self.classifier(x)

        return x


class ShadowLoader(DataLoader):
    """
    self.datas: {
        'node_offset': (num_nodes+1,),
        'edge_offset': (num_nodes+1,),
        'sub_nidx': (total_subg_nidx,),
        'sub_eidx': (2, total_subg_eidx),
        'ego_idx': (num_nodes,),
    }
    """
    def __init__(self, node_idx, egograph_datas, feat_matrix, labels, dim_label_smooth=0, **kwargs):
        self.dim_label_smooth = dim_label_smooth
        self.datas = egograph_datas
        self.X = feat_matrix
        self.y = labels.view(-1)
        if node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
        super().__init__(node_idx.tolist(), collate_fn=self.__collate__, **kwargs)

    def __collate__(self, batch_nodes):
        batch_data_list = []
        for bid in batch_nodes:
            subg_global_nids = self.datas['sub_nidx'][
                               self.datas['node_offset'][bid]: self.datas['node_offset'][bid + 1]]
            subg_edge_index  = self.datas['sub_eidx'][:,
                               self.datas['edge_offset'][bid]: self.datas['edge_offset'][bid + 1]]
            ego_data = Data(
                x=subg_global_nids,
                edge_index=subg_edge_index,
                # y=self.y[bid],
                ego_index=self.datas['ego_idx'][bid],
            )
            batch_data_list.append(ego_data)
        data = Batch.from_data_list(batch_data_list)
        data.x = self.X[data.x]
        data.y = self.y[batch_nodes]
        if self.dim_label_smooth > 0:
            data.x[data.ego_index, -self.dim_label_smooth:] = 0
        return data


def train(model, optimizer, metric, train_loader, epoch):
    model.train()
    metric.reset()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0
    for batch in train_loader:
        batch.to(metric.device)
        optimizer.zero_grad()

        if hasattr(batch, 'ego_index'):
            y_hat = model(batch.x, batch.edge_index, batch.ego_index, batch.ptr[:-1])
            batch_y = batch.y
        else:
            ego_index = torch.arange(batch.batch_size)
            y_hat = model(batch.x, batch.edge_index, ego_index)
            batch_y = batch.y[:batch.batch_size].squeeze()

        loss = F.cross_entropy(y_hat, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += float(loss)
        metric.update(y_hat.argmax(dim=-1), batch_y)
        pbar.update(batch_y.shape[0])
    pbar.close()

    return total_loss / len(train_loader), metric.compute()


@torch.no_grad()
def test(model, metric, *loaders):
    model.eval()
    ms = []
    for loader in loaders:
        metric.reset()
        pbar = tqdm(total=int(len(loader.dataset)))
        pbar.set_description(f'Evaluate: ')
        for data in loader:
            data.to(metric.device)

            if hasattr(data, 'ego_index'):
                y_hat = model(data.x, data.edge_index, data.ego_index, data.ptr[:-1])
                data_y = data.y
            else:
                ego_index = torch.arange(data.batch_size)
                y_hat = model(data.x, data.edge_index, ego_index)
                data_y = data.y[:data.batch_size].squeeze()

            metric.update(y_hat.argmax(dim=-1), data_y)
            pbar.update(data_y.shape[0])
        pbar.close()
        ms.append(metric.compute())
    return ms


@torch.no_grad()
def full_graph_inference(model, dataset, egograph_datas, device):
    inference_loader = ShadowLoader(np.arange(dataset['X'].shape[0]),
                                    egograph_datas,
                                    dataset['X'],
                                    dataset['y'],
                                    batch_size=4 * args.batch_size,
                                    num_workers=16,
                                    shuffle=False,
                                    persistent_workers=True)
    model.eval()
    pbar = tqdm(total=int(len(inference_loader.dataset)))
    pbar.set_description(f'Inference: ')
    y_hats = []
    for data in inference_loader:
        data.to(device)
        y_hat = model(data.x, data.edge_index, data.ego_index, data.ptr[:-1])
        y_hats.append(y_hat)
        pbar.update(data.y.shape[0])
    pbar.close()
    y_hats = torch.cat(y_hats, dim=0)
    return y_hats


def full_graph_evaluate(data_name, dataset, out):
    evaluator = Evaluator(name=data_name)
    train_idx = dataset['train_index']
    val_idx = dataset['valid_index']
    test_idx = dataset['test_index']
    pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y[train_idx],
        'y_pred': pred[train_idx]
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y[val_idx],
        'y_pred': pred[val_idx]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[test_idx],
        'y_pred': pred[test_idx]
    })['acc']
    return train_acc, val_acc, test_acc


def label_prop(dataset, alpha=0.2, itr_max=100, thres=0.015):
    train_idx = dataset['train_index']
    num_nodes, num_classes = dataset['X'].shape[0], dataset['y'].max().item() + 1

    signal = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    signal[train_idx] = F.one_hot(dataset['y'][train_idx].view(-1), num_classes).to(torch.float32)

    # symmetric normalization according to GCN formula
    I = dglsp.identity(shape=(og.num_nodes(), og.num_nodes()))
    A = dglsp.spmatrix(dataset['edge_index']) + I
    D = dglsp.diag(A.sum(dim=1))
    P = (D ** -0.5) @ A @ (D ** -0.5)
    P_t = P.t()

    # page rank to propagate training set labels
    print('Label Propagation...')
    H = signal
    Z = H
    for k in tqdm(range(itr_max)):
        Zk = (1 - alpha) * P_t @ Z + alpha * H
        delta_change = torch.linalg.norm(Z - Zk, ord='fro')
        Z = Zk
        if delta_change < thres:
            break
    print('Done!')

    signal_smoothed = Z
    signal_smoothed = torch.cat([signal, signal_smoothed], dim=1)
    return signal_smoothed


def cross_and_smooth(dataset, out_logits):
    y_soft = out_logits.softmax(dim=1).to('cpu')
    device = y_soft.device

    edge_index = dataset['edge_index']
    num_nodes = dataset['X'].shape[0]
    train_idx = dataset['train_index']
    y_train = dataset['y'][train_idx].to(device)

    adj_t = SparseTensor(row=edge_index[1], col=edge_index[0],
                         sparse_sizes=(num_nodes, num_nodes)).to(device)
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t

    print('Correct and smooth...')
    post = CorrectAndSmooth(num_correction_layers=50, correction_alpha=0.4,
                            num_smoothing_layers=50, smoothing_alpha=0.8,
                            autoscale=False, scale=20.)
    y_soft = post.correct(y_soft, y_train, train_idx, DAD)
    y_soft = post.smooth(y_soft, y_train, train_idx, DA)
    print('Done!')
    return y_soft


def main(args, dataset, egograph_datas, dim_label):
    num_features, num_classes = dataset['X'].shape[-1], dataset['y'].max().item() + 1

    if args.ns:
        d = PygNodePropPredDataset('ogbn-products', root='/data/gangda/ogb')
        split_idx = d.get_idx_split()
        data = d[0]
        train_loader = NeighborLoader(
            data,
            input_nodes=split_idx['train'],
            num_neighbors=[15, 10],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=12,
            persistent_workers=True,
        )
        val_loader = NeighborLoader(
            data,
            input_nodes=split_idx['valid'],
            num_neighbors=[15, 10],
            batch_size=args.batch_size,
            num_workers=12,
            persistent_workers=True,
        )
        test_loader = NeighborLoader(
            data,
            input_nodes=split_idx['test'],
            num_neighbors=[15, 10],
            batch_size=args.batch_size,
            num_workers=16,
            persistent_workers=True,
        )
    else:
        train_loader = ShadowLoader(dataset['train_index'],
                                    egograph_datas,
                                    dataset['X'],
                                    dataset['y'],
                                    dim_label_smooth=dim_label,  # mask ego nodes' labels
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    persistent_workers=True)
        val_loader = ShadowLoader(dataset['valid_index'],
                                  egograph_datas,
                                  dataset['X'],
                                  dataset['y'],
                                  batch_size=4 * args.batch_size,
                                  num_workers=8,
                                  persistent_workers=True)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    metric = Accuracy(task="multiclass", num_classes=num_classes)
    metric.to(device)

    checkpoint_path = osp.join(args.ckpt_path, 'checkpoint_{}_{}.tar'.format(os.getpid(), int(time.time())))
    out_logits_path = osp.join(args.ckpt_path, 'outlogits_{}_{}.pt'.format(os.getpid(), int(time.time())))

    # runs
    best_val, best_test = [], []
    for i in range(1, args.runs + 1):
        print(f'------------------------{i}------------------------')
        best_val_acc, best_epoch, test_acc = 0, 0, 0

        # seed per run
        seed_everything(i)

        # training
        model_kwargs = {'model': args.model,
                        'pooling': args.pool,
                        'norm': args.norm,
                        'dropout': args.dropout,
                        'dropedge': args.dropedge}
        if len(args.ckpt_file) > 0:
            if osp.isfile(args.ckpt_file):
                checkpoint_path = args.ckpt_file
            elif osp.isfile(osp.join(args.ckpt_path, args.ckpt_file)):
                checkpoint_path = osp.join(args.ckpt_path, args.ckpt_file)
            else:
                raise FileNotFoundError
            print('Found existing ckpt, skip training')
        elif len(args.out_file) > 0:
            if osp.isfile(args.out_file):
                out_logits_path = args.out_file
            elif osp.isfile(osp.join(args.ckpt_path, args.out_file)):
                out_logits_path = osp.join(args.ckpt_path, args.out_file)
            else:
                raise FileNotFoundError
            print('Found existing out_logits, skip training and testing')
        else:
            print('Start Training!')
            model = GNN(num_features, args.dim, num_classes, **model_kwargs).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
            for epoch in range(1, args.epochs + 1):
                loss, train_acc = train(model, optimizer, metric, train_loader, epoch)
                val_acc = test(model, metric, val_loader)[0]
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                    }, checkpoint_path)
                if epoch - best_epoch >= 10:
                    break
                print(f'Epoch: {epoch:02d}, Loss: {loss: .4f}, Approx. Train: {train_acc:.4f},'
                      f' Val: {val_acc:.4f}, Best Val: {best_val_acc:.4f}')
            print('Best model saved to path:', checkpoint_path)

        if args.ns:
            ckpt = torch.load(checkpoint_path)
            model = GNN(num_features, args.dim, num_classes, **model_kwargs).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            test_acc = test(model, metric, test_loader)[0]
            print(f"[Best Model] Epoch: {ckpt['epoch']:02d}, Train: {ckpt['train_acc']:.4f}, "
                  f"Val: {ckpt['val_acc']:.4f}, Test: {test_acc:.4f}")
        elif len(args.out_file) == 0:
            print('Start Testing!')
            ckpt = torch.load(checkpoint_path)
            model = GNN(num_features, args.dim, num_classes, **model_kwargs).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            out_logits = full_graph_inference(model, dataset, egograph_datas, device)
            train_acc, val_acc, test_acc = full_graph_evaluate(args.data_name, dataset, out_logits)
            print(f"[Best Model] Epoch: {ckpt['epoch']:02d}, Train: {ckpt['train_acc']:.4f}, "
                  f"Val: {ckpt['val_acc']:.4f}, Test: {test_acc:.4f}")
            torch.save(out_logits, out_logits_path)
            print('Output logits saved to path:', out_logits_path)

        if args.cs:
            print('Start Cross & Smooth')
            out_logits = torch.load(out_logits_path)
            out_soft = cross_and_smooth(dataset, out_logits)
            train_acc, val_acc, test_acc = full_graph_evaluate(args.data_name, dataset, out_soft)
            print(f"[After C&S] Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

        best_val.append(float(best_val_acc))
        best_test.append(float(test_acc))

    print(f'Valid: {np.mean(best_val):.4f} +- {np.std(best_val):.4f}')
    print(f'Test: {np.mean(best_test):.4f} +- {np.std(best_test):.4f}')


if __name__ == '__main__':
    args = parser.parse_args()

    # seed for data initialization
    seed_everything(0)

    # Load dataset
    dataset = None
    if args.data_name == 'ogbn-products':
        d = DglNodePropPredDataset(name='ogbn-products', root=args.dgl_path)
        og, y = d[0]
        dataset = dict(X=og.ndata['feat'], y=y)
        dataset['train_index'] = d.get_idx_split()['train']
        dataset['valid_index'] = d.get_idx_split()['valid']
        dataset['test_index'] = d.get_idx_split()['test']
        dataset['edge_index'] = torch.load(osp.join(args.data_path, 'dgl_edge_index.pt'))
        dataset['edge_index'] = add_remaining_self_loops(dataset['edge_index'])[0]
    else:
        raise NotImplementedError

    # feature standardization
    if args.stand:
        scaler = StandardScaler()
        scaler.fit(dataset['X'])
        X = scaler.transform(dataset['X'])
        dataset['X'] = torch.from_numpy(X).type(torch.get_default_dtype())

    # feature augmentation
    dim_label = 0
    if args.feat_aug:
        y_smoothed_path = osp.join(args.file_path, '{}_feature_smoothed.pt'.format(args.data_name))
        if osp.isfile(y_smoothed_path):
            y_smoothed = torch.load(y_smoothed_path)
        else:
            y_smoothed = label_prop(dataset, alpha=0.2, itr_max=100, thres=0.015)
            torch.save(y_smoothed, y_smoothed_path)
        dataset['X'] = torch.cat([dataset['X'], y_smoothed], dim=1)
        dim_label = y_smoothed.shape[1]

    # Load Ego-subgraph datas, execute load_shm.py before calling this func
    datas = fetch_datas_from_shm()

    print('Data Loading Complete!')

    main(args, dataset, datas, dim_label)
