import argparse
import time
import torch
from collections import defaultdict
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset


parser = argparse.ArgumentParser()
parser.add_argument('--max_degree', type=int, default=1000)
parser.add_argument('--drop_coe', type=float, default=0.9)
parser.add_argument('--num_source', type=float, default=100)
args = parser.parse_args()

lazy_alpha = 0.3
alpha = (2 * lazy_alpha) / (1 + lazy_alpha)
epsilon = 1e-6
top_k = 100

max_degree = args.max_degree
num_source = args.num_source
drop_coe = args.drop_coe

# Modify your test:
test_list = ['standard', 'drop']


def get_approx_ppr(key_, target_id_):
    ppr_func_dict = {
        'standard': standard_local_push_ppr,  # PPRGo, r_v = 0
        'uniform': uniform_local_push_ppr,  # same as 'standard'
        'batch': batch_local_push_ppr,  # update all source nodes simultaneously
        'bias': standard_local_push_ppr,  # MAPPR, 0 < r_v < threshold
        'lazy': standard_local_push_ppr,  # original local-push
        'lazy-uniform': uniform_local_push_ppr,
        'lazy-batch': batch_local_push_ppr,
        'clipped': standard_local_push_ppr,  # 'standard' ppr on clipped graph
        'clipped-lazy': standard_local_push_ppr,
        'clipped-batch': batch_local_push_ppr,
        'drop': standard_local_push_ppr,
        'drop-batch': batch_local_push_ppr,
    }
    kwargs = {
        'alpha_': lazy_alpha if 'lazy' in key_ else alpha,
        'gamma_': 0.5 if 'lazy' in key_ else 1.,
        'epsilon_': epsilon,
    }
    if 'bias' in key_:
        kwargs['beta_'] = 0.5
    if 'clipped' in key_:
        kwargs['max_degree_'] = max_degree
    if 'drop' in key_:
        kwargs['drop_coe_'] = drop_coe
    return ppr_func_dict[key_](g, target_id_, **kwargs)


def power_iter_ppr(P_w, target_id_, alpha_, epsilon_, max_iter, unweighted_degree_):
    num_nodes = P_w.size(0)
    s = torch.zeros(num_nodes)
    s[target_id_] = 1
    s = s.view(-1, 1)

    x = s.clone()
    num_push = torch.tensor([0])
    for i in range(max_iter):
        x_last = x
        x = alpha_ * s + (1 - alpha_) * (P_w @ x)
        # total num of operations
        x_last_nnz_index = x_last.view(-1).nonzero(as_tuple=False).view(-1)
        num_push += unweighted_degree_[x_last_nnz_index].sum()
        # check convergence, l1 norm
        if (abs(x - x_last)).sum() < num_nodes * epsilon_:
            print(f'power-iter      Iterations: {i}, Total Push Operations: {num_push.item()},'
                  f' NNZ: {(x.view(-1) > 0).sum()}')
            return x.view(-1), num_push.item()

    print(f'Failed to converge with tolerance({epsilon_}) and iter({max_iter})')
    return x.view(-1), num_push.item()


def standard_local_push_ppr(g_, target_id_, alpha_, epsilon_, beta_=0., gamma_=1., max_degree_=-1, drop_coe_=0.5):
    num_nodes = g_['weighted_degree'].size(-1)
    p = torch.zeros(num_nodes)
    r = torch.zeros(num_nodes)
    r[target_id_] = 1

    iterations, num_push = 0, torch.tensor([0])
    num_affected_nodes = num_affected_edges = 0
    threshold = epsilon_ * g_['weighted_degree']
    drop_threshold = drop_coe_ * threshold
    bias = beta_ * threshold

    while True:
        v_mask = r > threshold
        # if iterations == 15:
        #     break
        if v_mask.sum() == 0:
            break

        # drop nodes below specific threshold
        r[(r < drop_threshold) & (r > 0)] = 0

        v_idx = v_mask.nonzero(as_tuple=False).view(-1)
        for i, v in enumerate(v_idx):
            start, end = g_['indptr'][v], g_['indptr'][v + 1]
            num_degree = end - start
            if max_degree_ == -1 or num_degree <= max_degree_:
                ptr = torch.arange(start, end)
            else:
                ptr = torch.randperm(num_degree)[:max_degree_] + start
                num_affected_nodes += 1
                num_affected_edges += num_degree - max_degree_

            u_idx = g_['indices'][ptr]
            u_weights = g_['edge_weights'][ptr]

            while r[v] > threshold[v]:
                # update source node
                if beta_ == 0:
                    p[v] += alpha_ * r[v]
                    m_v = (1 - alpha_) * gamma_ * r[v]
                    r[v] = (1 - alpha_) * (1 - gamma_) * r[v]
                else:
                    p[v] += alpha_ * (r[v] - bias[v])
                    m_v = (1 - alpha_) * (r[v] - bias[v])
                    r[v] = bias[v]

                # batch update neighbors
                r[u_idx] += m_v * (u_weights / u_weights.sum())

                num_push += end - start

        iterations += 1

    return p, iterations, num_push.item(), num_affected_nodes, num_affected_edges


def uniform_local_push_ppr(g_, target_id_, alpha_, epsilon_, gamma_=1.):
    num_nodes = g_['weighted_degree'].size(-1)
    p = torch.zeros(num_nodes)
    r = torch.zeros(num_nodes)
    r[target_id_] = 1

    iterations, num_push = 0, torch.tensor([0])
    threshold = epsilon_ * g_['weighted_degree']
    while True:
        v_mask = r > threshold
        if v_mask.sum() == 0:
            break

        v_idx = v_mask.nonzero(as_tuple=False).view(-1)
        for i, v in enumerate(v_idx):
            # update source node
            p[v] += alpha_ * r[v]
            temp = (1 - alpha_) * r[v]
            r[v] = (1 - gamma_) * temp
            m_v = gamma_ * temp / g_['weighted_degree'][v]

            # batch update neighbors
            start, end = g_['indptr'][v], g_['indptr'][v + 1]
            u_idx = g_['indices'][start: end]
            u_weights = g_['edge_weights'][start: end]
            r[u_idx] += m_v * u_weights

            num_push += end - start

        iterations += 1

    return p, iterations, num_push.item(), 0, 0


def batch_local_push_ppr(g_, target_id_, alpha_, epsilon_, gamma_=1., max_degree_=-1, drop_coe_=0.5):
    num_nodes = g_['weighted_degree'].size(-1)
    p = torch.zeros(num_nodes)
    r = torch.zeros(num_nodes)
    r[target_id_] = 1

    iterations, num_push = 0, torch.tensor([0])
    num_affected_nodes = num_affected_edges = 0
    threshold = epsilon_ * g_['weighted_degree']
    drop_threshold = drop_coe_ * threshold

    while True:
        v_mask = r > threshold
        if v_mask.sum() == 0:
            break

        # drop nodes below specific threshold
        r[(r < drop_threshold) & (r > 0)] = 0

        v_idx = v_mask.nonzero(as_tuple=False).view(-1)
        while True:
            v_mask_ = r[v_idx] > threshold[v_idx]
            if v_mask_.sum() == 0:
                break
            v_idx_ = v_idx[v_mask_]

            # batch update source nodes
            p[v_idx_] += alpha_ * r[v_idx_]
            m = (1 - alpha_) * r[v_idx_]
            r[v_idx_] = (1 - gamma_) * m
            m = gamma_ * m

            for i, v in enumerate(v_idx_):
                start, end = g_['indptr'][v], g_['indptr'][v + 1]
                num_degree = end - start
                if max_degree_ == -1 or num_degree <= max_degree_:
                    ptr = torch.arange(start, end)
                else:
                    ptr = torch.randperm(num_degree)[:max_degree_] + start
                    num_affected_nodes += 1
                    num_affected_edges += num_degree - max_degree_
                # batch update neighbors
                u_idx = g_['indices'][ptr]
                u_weights = g_['edge_weights'][ptr]
                r[u_idx] += m[i] * u_weights / u_weights.sum()

                num_push += end - start

        iterations += 1

    return p, iterations, num_push.item(), num_affected_nodes, num_affected_edges


if __name__ == '__main__':
    """Graph Loading"""
    dataset = PygNodePropPredDataset(name='ogbn-products', root='/data/gangda/ogb', transform=T.ToSparseTensor())
    # dataset = Planetoid(root='/data/gangda/pyg', name='Cora', pre_transform=T.ToSparseTensor())
    data = dataset[0]
    unweighted_degree = data.adj_t.sum(dim=0).to(torch.int)
    coo_adj_edge_weight = torch.rand(data.num_edges)
    data.adj_t.set_value_(coo_adj_edge_weight, layout='csc')

    # local-push
    indptr, indices, value = data.adj_t.csc()
    degree = data.adj_t.sum(dim=0).to(torch.float)
    g = {
        'indptr': indptr,
        'indices': indices,
        'edge_weights': value,
        'weighted_degree': degree,
    }

    # power-iteration
    norm_adj_t = data.adj_t * degree.pow(-1).view(1, -1)


    """Testing"""
    total_base_p = 0.
    total_concur, total_err, total_push = defaultdict(int), defaultdict(float), defaultdict(int)
    total_clipped_nodes, total_clipped_edges = defaultdict(int), defaultdict(int)
    total_time = defaultdict(float)

    source_nodes = torch.randperm(data.num_nodes)[:num_source]
    for epoch, target_id in enumerate(source_nodes):
        print(f'\n########## Iter {epoch + 1} ##########')

        tik = time.time()
        base_p, base_num_push = power_iter_ppr(norm_adj_t, target_id, alpha, 1e-10, 100, unweighted_degree)
        total_time['power-iter'] += time.time() - tik

        total_base_p += base_p
        total_push['power-iter'] += base_num_push
        _, base_top_k = torch.sort(base_p, descending=True)

        for key in test_list:
            tik = time.time()
            approx_p, approx_num_iter, approx_num_push, num_clipped_nodes, num_clipped_edges = \
                get_approx_ppr(key, target_id)
            total_time[key] += time.time() - tik

            print(f'{key:15s} Iterations: {approx_num_iter}, Total Push Operations: {approx_num_push},'
                  f' NNZ: {(approx_p > 0).sum()}, Clipped Nodes(Edges): {num_clipped_nodes}({num_clipped_edges})')

            total_push[key] += approx_num_push
            total_err[key] += (abs(approx_p - base_p)).sum().item()
            _, approx_top_k = torch.sort(approx_p, descending=True)
            total_concur[key] += np.intersect1d(base_top_k[:top_k], approx_top_k[:top_k]).shape[0]

            total_clipped_nodes[key] += num_clipped_nodes
            total_clipped_edges[key] += num_clipped_edges


    """Logging"""
    print(f'\n\n########## Results ##########')
    print(f'Parameters: lazy_alpha={lazy_alpha}, alpha={alpha:.3f}, epsilon={epsilon:.1e}, max_degree={max_degree},'
          f' drop_coe={drop_coe}')

    print(f'\nAvg Push Operations:')
    for k, val in total_push.items():
        print(f'\t{k}: {val / num_source:.0f}')

    print(f'\nAvg Clipped Nodes(Edges):')
    for k, val in total_clipped_nodes.items():
        print(f'\t{k}: {val / num_source:.0f}({total_clipped_edges[k] / num_source:.0f})')

    print(f'\nPrecision Top-{top_k}:')
    for k, val in total_concur.items():
        print(f'\t{k}: {val / (top_k * num_source):.3f}')

    print(f'\nmean power-iter ppr: {total_base_p.mean().item() / num_source: .3e}'
          f'\nMean Absolute Error:')
    for k, val in total_err.items():
        print(f'\t{k}: {val / (data.num_nodes * num_source):.3e}')

    print(f'\nAvg Run Time:')
    for k, val in total_time.items():
        print(f'\t{k}: {val / num_source:.3f}s')