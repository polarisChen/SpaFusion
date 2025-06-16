# -*- coding:utf-8 -*-
"""
Author：polaris
Data：2024年06月25日
"""

import torch
import numpy as np
import random
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from sklearn.cluster import KMeans
from evaluate import eva
import torch.nn.functional as F
import os
import pickle
from sklearn.cluster import DBSCAN


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_adj_matrices(adj, spatial_path1, spatial_path2, feature1_path, feature2_path):
    """Save adjacency matrices to specified paths."""
    with open(spatial_path1, 'wb') as f:
        pickle.dump(adj['adj_spatial_omics1'], f)

    with open(spatial_path2, 'wb') as f:
        pickle.dump(adj['adj_spatial_omics2'], f)
    
    with open(feature1_path, 'wb') as f:
        pickle.dump(adj['adj_feature_omics1'], f)
    
    with open(feature2_path, 'wb') as f:
        pickle.dump(adj['adj_feature_omics2'], f)


def load_adj_matrices(spatial_path1, spatial_path2, feature1_path, feature2_path):
    """Load adjacency matrices from specified paths."""
    with open(spatial_path1, 'rb') as f:
        spatial_adj1 = pickle.load(f)

    with open(spatial_path2, 'rb') as f:
        spatial_adj2 = pickle.load(f)
    
    with open(feature1_path, 'rb') as f:
        adj_feature_omics1 = pickle.load(f)
    
    with open(feature2_path, 'rb') as f:
        adj_feature_omics2 = pickle.load(f)

    adj = {
        'adj_spatial_omics1': spatial_adj1,
        'adj_spatial_omics2': spatial_adj2,
        'adj_feature_omics1': adj_feature_omics1,
        'adj_feature_omics2': adj_feature_omics2,
    }
    return adj


def adjacent_matrix_preprocessing(adata_omics1, adata_omics2, adj_path):
    # 文件路径
    spatial_path1 = os.path.join(adj_path, 'adj_spatial_omics1.npy')
    spatial_path2 = os.path.join(adj_path, 'adj_spatial_omics2.npy')
    feature1_path = os.path.join(adj_path, 'adj_feature_omics1.npy')
    feature2_path = os.path.join(adj_path, 'adj_feature_omics2.npy')

    if all(os.path.exists(path) for path in [spatial_path1, spatial_path2, feature1_path, feature2_path]):
        print("Loading Adj Matrix...")
        adj_spatial_omics1 = np.load(spatial_path1)
        adj_spatial_omics2 = np.load(spatial_path2)
        adj_feature_omics1 = np.load(feature1_path)
        adj_feature_omics2 = np.load(feature2_path)
    else:

        print("construct Adj Matrix...")
        # construct spatial graph
        adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
        adj_spatial_omics1 = construct_graph(adj_spatial_omics1)
        adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
        adj_spatial_omics2 = construct_graph(adj_spatial_omics2)

        adj_spatial_omics1 = adj_spatial_omics1.toarray()  # To ensure that adjacent matrix is symmetric
        adj_spatial_omics2 = adj_spatial_omics2.toarray()

        adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
        adj_spatial_omics1 = np.where(adj_spatial_omics1 > 1, 1, adj_spatial_omics1)
        adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
        adj_spatial_omics2 = np.where(adj_spatial_omics2 > 1, 1, adj_spatial_omics2)

        # construct feature graph
        adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
        adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())

        adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
        adj_feature_omics1 = np.where(adj_feature_omics1 > 1, 1, adj_feature_omics1)
        adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
        adj_feature_omics2 = np.where(adj_feature_omics2 > 1, 1, adj_feature_omics2)

        # saving adj matrix
        np.save(spatial_path1, adj_spatial_omics1)
        np.save(spatial_path2, adj_spatial_omics2)
        np.save(feature1_path, adj_feature_omics1)
        np.save(feature2_path, adj_feature_omics2)

    adj = {
        'adj_spatial_omics1': adj_spatial_omics1,
        'adj_spatial_omics2': adj_spatial_omics2,
        'adj_feature_omics1': adj_feature_omics1,
        'adj_feature_omics2': adj_feature_omics2,
        }

    return adj


def construct_graph(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def target_distribution(Q):
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P


# clustering guidance
def distribution_loss(Q, P):
    
    loss = F.kl_div((Q[0].log() + Q[1].log() + Q[2].log()) / 3, P, reduction='batchmean')
    
    # loss = F.kl_div(Q[0].log(), P, reduction='batchmean')

    return loss


def assignment(Q, y):
    y_pred = torch.argmax(Q, dim=1).data.cpu().numpy()
    if y is not None:
        acc, f1, nmi, ari, ami, vms, fms = eva(y, y_pred)
        return acc, f1, nmi, ari, ami, vms, fms, y_pred
    else:
        return None, None, None, None, None, None, None, y_pred

#
# def clustering(Z, y):
#     # if y != []:
#     model = KMeans(n_clusters=len(np.unique(y)), n_init=10)
#     cluster_id = model.fit_predict(Z.data.cpu().numpy())
#
#     # acc, f1, nmi, ari, ami, vms, fms = eva(y, cluster_id)
#
#     return model.cluster_centers_
#
#     # else:
#     #     model = KMeans(n_clusters=10, n_init=10)
#     #     cluster_id = model.fit_predict(Z.data.cpu().numpy())
#     #     return model.cluster_centers_


def clustering(Z, y=None, n_clusters=None):
    if y is not None and len(y) > 0:
        model = KMeans(n_clusters=len(np.unique(y)), n_init=10)
    else:
        if n_clusters is None:
            raise ValueError("n_clusters must be specified when y is None or empty.")
        model = KMeans(n_clusters=n_clusters, n_init=10)
    cluster_id = model.fit_predict(Z.data.cpu().numpy())

    return model.cluster_centers_


def laplacian_kernel_similarity(A, alpha):
    """
    计算矩阵 A 中每一对细胞（行）之间的 Laplacian kernel 相似度。
    """
    num_cells = A.shape[1]
    Sc = np.zeros((num_cells, num_cells))
    for i in range(num_cells):
        for j in range(num_cells):
            distance = np.linalg.norm(A[:, i] - A[:, j])
            Sc[i, j] = np.exp(-distance / alpha)
    return Sc


def weighted_p_NKN(A, Sc, p, rho):
    """
    使用加权 p-NKN 算法优化 scRNA-seq 数据。
    """
    A_dot = (A > 0).astype(int)

    row_A_dot, col_A_dot = A_dot.shape
    A_d_delta = np.zeros((row_A_dot, col_A_dot))

    for i in range(col_A_dot):
        sort_indices = np.argsort(Sc[:, i])[::-1]
        p_adjusted = min(p, len(sort_indices))
        Q = np.sum(Sc[sort_indices[:p_adjusted], i])
        for j in range(p_adjusted):
            beta_i = (rho ** j) * Sc[sort_indices[j], i]
            A_d_delta[:, i] += beta_i * A_dot[:, sort_indices[j]]
        A_d_delta[:, i] /= Q

    A_OPT = np.maximum(A_dot, A_d_delta)
    return A_OPT


def regularization_loss(X):
    z = torch.mm(X.t(), X)
    squared = (torch.diag(z) - 1) ** 2
    regularization_loss = torch.sum(squared)

    return regularization_loss