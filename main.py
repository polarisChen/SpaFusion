# -*- coding:utf-8 -*-
"""
Author：polaris
Data：2024年06月25日
"""
import torch

from processing import *
from utils import *
from encoder import *
from high_order_matrix import process_adjacency_matrix
from evaluate import *
import torch.optim as optim
import time
import argparse
from copy import deepcopy

import os

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pre_train(x1, x2, spatial_adj1, feature_adj1, spatial_adj2, feature_adj2, Mt1, Mt2, y, n_clusters, num_epoch, device, weight_list, lr):
    model = GCNAutoencoder(input_dim1=x1.shape[1], input_dim2=x2.shape[1], enc_dim1=256, enc_dim2=128, dec_dim1=128,
                           dec_dim2=256, latent_dim=20, dropout=0.1, num_layers=2, num_heads1=1, num_heads2=1,
                           n_clusters=n_clusters, n_node=x1.shape[0])

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pretrain_loss = []
    for epoch in range(num_epoch):
        Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, _ = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, pretrain=True)

        loss_ae1 = F.mse_loss(a11_hat, spatial_adj1)
        loss_ae2 = F.mse_loss(a12_hat, feature_adj1)
        loss_ae3 = F.mse_loss(a21_hat, spatial_adj2)
        loss_ae4 = F.mse_loss(a22_hat, feature_adj2)

        loss_x1 = F.mse_loss(x13_hat, x1)
        loss_x2 = F.mse_loss(x23_hat, x2)

        loss_rec = weight_list[0] * loss_ae1 + weight_list[1] * loss_ae2 + weight_list[2] * loss_ae3 + weight_list[3] * loss_ae4 + weight_list[4] * loss_x1 + weight_list[5] * loss_x2

        loss = loss_rec
 
        pretrain_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch: {:.0f}/{:.0f} ,loss:{:.8f}".format(epoch + 1, num_epoch, loss))

    torch.save(model.state_dict(), r'./pretrain/{}_pre_model.pkl'.format(opt.name))
    # np.save(r"./loss/{}_pre_train_loss.npy".format(opt.name), np.array(pretrain_loss))
    return z1_tilde, z2_tilde


def train(x1, x2, spatial_adj1, feature_adj1, spatial_adj2, feature_adj2, Mt1, Mt2, y, n_clusters, num_epoch, lambda1, device, seed, lambda2, weight_list, lr, num, spatial_K, adj_K):
    model = GCNAutoencoder(input_dim1=x1.shape[1], input_dim2=x2.shape[1], enc_dim1=256, enc_dim2=128, dec_dim1=128,
                           dec_dim2=256, latent_dim=20, dropout=0.1, num_layers=2, num_heads1=1, num_heads2=1,
                           n_clusters=n_clusters, n_node=x1.shape[0])
    
    model.to(device)

    # loading pretrained model
    model.load_state_dict(torch.load(r'./pretrain/{}_pre_model.pkl'.format(opt.name), map_location='cpu'))

    with torch.no_grad():
        Z, z1_tilde, z2_tilde, _, _, _, _, _, _, _ = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2)

    centers1 = clustering(Z, y, n_clusters=n_clusters)  

    # initialize cluster centers
    model.cluster_centers1.data = torch.tensor(centers1).to(device)

    train_losses = []
    ari_ = []
    optimizer = optim.Adam(model.parameters(), lr=lr) 

    for epoch in range(num_epoch):
        Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, Q = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, pretrain=False)
        loss_ae1 = F.mse_loss(a11_hat, spatial_adj1)
        loss_ae2 = F.mse_loss(a12_hat, feature_adj1)
        loss_ae3 = F.mse_loss(a21_hat, spatial_adj2)
        loss_ae4 = F.mse_loss(a22_hat, feature_adj2)
        loss_x1 = F.mse_loss(x13_hat, x1)
        loss_x2 = F.mse_loss(x23_hat, x2)
        dense_loss1 = torch.mean((Z - z1_tilde) ** 2)
        dense_loss2 = torch.mean((Z - z2_tilde) ** 2)
        loss_rec = weight_list[0] * loss_ae1 + weight_list[1] * loss_ae2 + weight_list[2] * loss_ae3 + weight_list[3] * loss_ae4 + weight_list[4] * loss_x1 + weight_list[5] * loss_x2
        L_KL1 = distribution_loss(Q, target_distribution(Q[0].data))
        loss = loss_rec + lambda1 * L_KL1 + lambda2 * (dense_loss1 + dense_loss2)

        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch: {:.0f}/{:.0f} ,loss:{:.8f}".format(epoch + 1, num_epoch, loss))
   
        # clustering & evaluation
        if y is not None:
            acc, f1, nmi, ari, ami, vms, fms, y_pred = assignment((Q[0]).data, y)
        else:
            y_pred = torch.argmax(Q[0].data, dim=1).data.cpu().numpy()

    # saving results……
    result_dir = './results/{}'.format(opt.name)
    os.makedirs(result_dir, exist_ok=True)

    if y is not None:
        with open(os.path.join(result_dir, '{}_performance.csv'.format(opt.name)), 'a') as f:
            f.write("seed:{}, lambda1:{}, lambda2:{}, spatial_k:{}, adj_k:{}, wieght_list:{}, ".format(seed, lambda1, lambda2, spatial_K, adj_K, weight_list))
            f.write('%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (acc, f1, nmi, ari, ami, vms, fms))
    else:
        pass

    np.save(os.path.join(result_dir, '{}_{}_pre_label.npy'.format(opt.name, num)), y_pred)
    np.save(os.path.join(result_dir, '{}_{}_laten.npy'.format(opt.name, num)), Z.data.cpu().numpy())

    return z1_tilde, z2_tilde


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Model setting……")
    parser.add_argument('--name', type=str, default='D1', help='dataset name')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--spatial_k', type=int, default=9, help='spatial_k')
    parser.add_argument('--adj_k', type=int, default=20, help='adj_k')
    parser.add_argument('--lambda1', type=float, default=1, help='lambda1')
    parser.add_argument('--lambda2', type=float, default=0.1, help='lambda2')
    parser.add_argument('--weight_list', type=list, default=[1, 1, 1, 1, 1, 1], help='weight list')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--pretrain_epoch', type=int, default=5000, help='pretrain epoch')
    parser.add_argument('--train_epoch', type=int, default=2500, help='train epoch')

    opt = parser.parse_args()

    print("setting:")
    print("------------------------------")
    print("dataset        : {}".format(opt.name))
    print("device         : {}".format(opt.device))
    print("seed           : {}".format(opt.seed))
    print("spatial_k      : {}".format(opt.spatial_k))
    print("adj_k          : {}".format(opt.adj_k))
    print("lambda1        : {}".format(opt.lambda1))
    print("lambda2        : {}".format(opt.lambda2))
    print("weight_list    : {}".format(opt.weight_list))
    print("learning rate  : {:.0e}".format(opt.lr))
    print("pretrain epoch : {}".format(opt.pretrain_epoch))
    print("training epoch : {}".format(opt.train_epoch))
    print("------------------------------")
    setup_seed(opt.seed)

    # read data
    data_path = "data/"
    labels = pd.read_csv(data_path + 'D1_annotation_labels.csv')
    label = labels['labels']

    if label is not None:
        n_clusters = len(np.unique(label))  
    else:
        n_clusters = 5

    adata_omics1 = sc.read_h5ad(data_path + 'adata_RNA.h5ad')
    adata_omics2 = sc.read_h5ad(data_path + 'adata_ADT.h5ad')
    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()
    adata_omics1, adata_omics2 = load_data(adata_omics1=adata_omics1, view1="RNA", adata_omics2=adata_omics2, view2="Protein", 
                                            n_neighbors=opt.spatial_k, k=opt.adj_k)
    
    # feature matrix
    data1 = adata_omics1.obsm['feat'].copy()
    data2 = adata_omics2.obsm['feat'].copy()

    # graph
    adj_path = "./pre_adj/{}".format(opt.name)
    os.makedirs(adj_path, exist_ok=True)
    adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2, adj_path)

    # feature graph
    feature_adj1 = adj['adj_feature_omics1']
    feature_adj2 = adj['adj_feature_omics2']
    # spatial graph
    spatial_adj1 = adj['adj_spatial_omics1']
    spatial_adj2 = adj['adj_spatial_omics2']

    # high-order graph
    Mt1 = process_adjacency_matrix(feature_adj1, "./pre_adj/{}/{}_Mt1.npy".format(opt.name, opt.name))
    Mt2 = process_adjacency_matrix(feature_adj2, "./pre_adj/{}/{}_Mt2.npy".format(opt.name, opt.name))

    def is_symmetric(matrix, tol=1e-8):
        return np.allclose(matrix, matrix.T, atol=tol)

    feature_adj1 = norm_adj(feature_adj1)
    feature_adj2 = norm_adj(feature_adj2)
    spatial_adj1 = norm_adj(spatial_adj1)
    spatial_adj2 = norm_adj(spatial_adj2)
    Mt1 = norm_adj(Mt1)
    Mt2 = norm_adj(Mt2)
    data1 = torch.tensor(data1, dtype=torch.float32).to(device)
    data2 = torch.tensor(data2, dtype=torch.float32).to(device)
    feature_adj1 = torch.tensor(feature_adj1, dtype=torch.float32).to(device)
    feature_adj2 = torch.tensor(feature_adj2, dtype=torch.float32).to(device)
    spatial_adj1 = torch.tensor(spatial_adj1, dtype=torch.float32).to(device)
    spatial_adj2 = torch.tensor(spatial_adj2, dtype=torch.float32).to(device)
    Mt1 = torch.tensor(Mt1, dtype=torch.float32).to(device)
    Mt2 = torch.tensor(Mt2, dtype=torch.float32).to(device)

    # Abaltion
    spatial_adj1 = spatial_adj1 * feature_adj1
    spatial_adj2 = spatial_adj2 * feature_adj2

    print("============dataset shape=================")
    print("n_clusters:{}".format(n_clusters))
    print("data1.shape:{}".format(data1.shape))
    print("data1.feature.shape:{}".format(feature_adj1.shape))
    print("data1.highOrder.shape:{}".format(Mt1.shape))

    print("================================Pre_training...============================================")
    z1_tilde, z2_tilde = pre_train(x1=data1, x2=data2, spatial_adj1=spatial_adj1, feature_adj1=feature_adj1,
                                spatial_adj2=spatial_adj2, feature_adj2=feature_adj2, Mt1=Mt1, Mt2=Mt2, y=label, n_clusters=n_clusters,
                                num_epoch=opt.pretrain_epoch, device=device, weight_list=opt.weight_list, lr=opt.lr)

    for i in range(10):
        print("================================Training... {}============================================".format(i))
        z1_tilde, z2_tilde = train(x1=data1, x2=data2, spatial_adj1=spatial_adj1, feature_adj1=feature_adj1, spatial_adj2=spatial_adj2,
                                feature_adj2=feature_adj2, y=label, n_clusters=n_clusters, Mt1=Mt1, Mt2=Mt2, num_epoch=opt.train_epoch, lambda1=opt.lambda1,
                                device=device, seed=opt.seed, lambda2=opt.lambda2, weight_list=opt.weight_list, lr=opt.lr, num=i, 
                                spatial_K=opt.spatial_k, adj_K=opt.adj_k)
    print("======= Finish ==========")



