"""
Some functions are adapted from previous studies: https://github.com/AI4HealthUOL/SSSD, https://github.com/tsy935/graphs4mer/tree/main

"""

import os
import numpy as np
import torch
import torch_geometric
import random
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, recall_score, precision_score, auc
from collections import defaultdict
import dgl

def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


# Utilities for Decontaminator

def std_normal(size):
    return torch.normal(0, 1, size=size).cuda()


def compute_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def compute_diffusion_hyperparamters(T, beta_0, beta_T):

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \bar{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1}) (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \bar{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams



def mask_RandM(sample, k):

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    for channel in range(mask.shape[1]):
        perm = torch.randperm(len(length_index))
        idx = perm[0:k]
        mask[:, channel][idx] = 0

    return mask


def mask_RandBM(sample, k):

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


def mask_BoM(sample, k):

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


# Utilities for Variable Dependency Modeling

def knn_graph(x, k, dist_measure="cosine", undirected=True):
    if dist_measure == "euclidean":
        dist = torch.cdist(x, x, p=2.0)
        dist = (dist - dist.min()) / (dist.max() - dist.min())
        knn_val, knn_ind = torch.topk(
            dist, k, dim=-1, largest=False
        )  # smallest distances
    elif dist_measure == "cosine":
        norm = torch.norm(x, dim=-1, p="fro")[:, :, None]
        x_norm = x / norm
        dist = torch.matmul(x_norm, x_norm.transpose(1, 2))
        knn_val, knn_ind = torch.topk(
            dist, k, dim=-1, largest=True
        )  # largest similarities
    else:
        raise NotImplementedError

    adj_mat = (torch.ones_like(dist) * 0).scatter_(-1, knn_ind, knn_val).to(x.device)

    adj_mat = torch.clamp(adj_mat, min=0.0)  # remove negatives

    if undirected:
        adj_mat = (adj_mat + adj_mat.transpose(1, 2)) / 2

    # add self-loop
    I = (
        torch.eye(adj_mat.shape[-1], adj_mat.shape[-1])
            .unsqueeze(0)
            .repeat(adj_mat.shape[0], 1, 1)
            .to(bool)
    ).to(x.device)
    adj_mat = adj_mat * (~I) + I

    # to sparse graph
    edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_mat)

    return edge_index, edge_weight, adj_mat


def prune_graph(adj_mat, num_nodes, method="thresh", edge_top_perc=None, knn=None, thresh=None):
    if method == "thresh":
        sorted, indices = torch.sort(
            adj_mat.reshape(-1, num_nodes * num_nodes),
            dim=-1,
            descending=True,
        )
        K = int((num_nodes ** 2) * edge_top_perc)
        mask = adj_mat > sorted[:, K].unsqueeze(1).unsqueeze(2)
        adj_mat = adj_mat * mask
    elif method == "knn":
        knn_val, knn_ind = torch.topk(
            adj_mat, knn, dim=-1, largest=True
        )
        adj_mat = (torch.ones_like(adj_mat) * 0).scatter_(-1, knn_ind, knn_val).to(adj_mat.device)
    elif method == "thresh_abs":
        mask = (adj_mat > thresh).float()
        adj_mat = adj_mat * mask
    else:
        raise NotImplementedError

    return adj_mat


def calculate_normalized_laplacian(adj):
    batch, num_nodes, _ = adj.shape
    d = adj.sum(-1)  # (batch, num_nodes)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # (batch, num_nodes, num_nodes)

    identity = (torch.eye(num_nodes).unsqueeze(0).repeat(batch, 1, 1)).to(
        adj.device
    )  # (batch, num_nodes, num_nodes)
    normalized_laplacian = identity - torch.matmul(
        torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt
    )

    return normalized_laplacian


def feature_smoothing(adj, X):
    # normalized laplacian
    L = calculate_normalized_laplacian(adj)

    feature_dim = X.shape[-1]
    mat = torch.matmul(torch.matmul(X.transpose(1, 2), L), X) / (feature_dim ** 2)
    loss = mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    return loss

def aug(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.num_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)

    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    return ng, feat


def drop_feature(x, drop_prob):
    drop_mask = (
        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

def dict_metrics(y_pred, y, y_prob=None, file_names=None, average='binary'):

    metrics_dict = {}
    predicted_dict = defaultdict(list)
    true_dict = defaultdict(list)

    # write into output dictionary
    if file_names is not None:
        for i, file_name in enumerate(file_names):
            predicted_dict[file_name] = y_pred[i]
            true_dict[file_name] = y[i]

    if y is not None:
        metrics_dict['F1'] = f1_score(y_true=y, y_pred=y_pred, average=average)
        metrics_dict['precision'] = precision_score(y_true=y, y_pred=y_pred, average=average)
        metrics_dict['recall'] = recall_score(y_true=y, y_pred=y_pred, average=average)
        precision, recall, thresholds = precision_recall_curve(y, y_pred)
        # Use AUC function to calculate the area under the curve of precision recall curve
        auc_precision_recall = auc(recall, precision)
        metrics_dict["aucpr"] = auc_precision_recall
    return metrics_dict
