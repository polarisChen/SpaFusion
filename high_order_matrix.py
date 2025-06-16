# -*- coding:utf-8 -*-
"""
Author：polaris
Data：2024年07月29日
"""

import os
import time
import numpy as np


def find_3_node_motifs(adjacency_matrix):
    motifs = []
    num_nodes = adjacency_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adjacency_matrix[i, j]:
                for k in range(j+1, num_nodes):
                    if adjacency_matrix[i, k] and adjacency_matrix[j, k]:
                        motifs.append((i, j, k))
    return motifs


def save_matrix(matrix, filename):
    np.save(filename, matrix)


def load_matrix(filename):
    return np.load(filename)


def process_adjacency_matrix(adjacency_matrix, filename):
    if os.path.exists(filename):
        print(f"load Mt……：{filename}")
        return load_matrix(filename)
    else:
        start_time = time.time()
        three_node_motifs = find_3_node_motifs(adjacency_matrix=adjacency_matrix)
        end_time = time.time()
        time_ = end_time - start_time
        print("3-node motifs:{}, time costing:{}".format(len(three_node_motifs), time_))
        num_nodes = adjacency_matrix.shape[0]
        Mt = np.zeros((num_nodes, num_nodes), dtype=int)
        for i, j, k in three_node_motifs:
            Mt[i][j] += 1
            Mt[i][k] += 1
            Mt[j][k] += 1
            Mt[j][i] += 1
            Mt[k][i] += 1
            Mt[k][j] += 1
        save_matrix(Mt, filename)
        return Mt

