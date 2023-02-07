import torch


# for i, v in enumerate(v_idx):
#     v_host_id = 0
#     while v_host_id < NUM_MACHINES:
#         if cluster_ptr[v_host_id] <= v < cluster_ptr[v_host_id+1]:
#             break
#         v_host_id += 1
#
#     u_idx, u_weights = graph_rrefs[v_host_id].rpc_sync().fetch_neighbor_list(v - cluster_ptr[v_host_id])
#     print(v_host_id, u_idx)
#     v_degree = u_weights.sum()
#     r[u_idx] += m_v[i] * u_weights / v_degree
#
#     # update sampled degree
#     visited_degrees[v] = v_degree

