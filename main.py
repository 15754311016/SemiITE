import time
import argparse
import numpy as np
# a
import torch
# import torch.nn.functional as F
import torch.optim as optim
import trinet
import train
from train import wasserstein
from sklearn import preprocessing
from choose import choose_data
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
loss = nn.MSELoss()
np.random.seed(43)
torch.manual_seed(31)

def load_ihdp(path):
    data = np.loadtxt(open(path, "rb"), delimiter=",")
    X = data[:, 5:]
    X = preprocessing.scale(X)
    t = data[:, 0:1].reshape(1, -1)
    yf = data[:, 1:2].reshape(1, -1)
    y_cf = data[:, 2:3].reshape(1, -1)
    Y1 = np.where(t > 0, yf, y_cf)
    Y0 = np.where(t > 0, y_cf, yf)
    Y1 = Tensor(np.squeeze(Y1))
    Y0 = Tensor(np.squeeze(Y0))
    t = LongTensor(np.squeeze(t))
    X = Tensor(X)
    return X, t, Y1, Y0


def split_L_U(X, T, Y1, Y0, u_ratio=0.6, L_ratio=0.2, test_ratio=0.2):
    N = X.shape[0]
    idx = np.random.permutation(N)
    n_labeled = int(N * L_ratio)
    n_unlabeled = int(N * u_ratio)
    n_test = int(N * test_ratio)
    idx_label, idx_unlabel, idx_test = idx[:n_labeled], idx[n_labeled:n_labeled + n_unlabeled], idx[
                                                                                                n_labeled + n_unlabeled:]
    X_train_L = X[idx_label]
    T_train_L = T[idx_label]
    Y1_train_L, Y0_train_L = Y1[idx_label], Y0[idx_label]
    X_train_U = X[idx_unlabel]
    T_train_U = T[idx_unlabel]
    Y1_train_U, Y0_train_U = Y1[idx_unlabel], Y0[idx_unlabel]
    X_test = X[idx_test]
    T_test = T[idx_test]
    Y1_test, Y0_test = Y1[idx_test], Y0[idx_test]
    return X_train_L, Y1_train_L, Y0_train_L, X_train_U, Y1_train_U, Y0_train_U, X_test, Y1_test, Y0_test, T_train_L, T_train_U, T_test


def eval(model, X, T, Y1, Y0,update=False):
    model.eval()
    if update==True:
        yf_pred, _ = model(X, T, 1)
        ycf_pred, _ = model(X, 1 - T, 1)
        #print("y1_pred:", yf_pred.shape)
        y1_pred, y0_pred = torch.where(T > 0, yf_pred, ycf_pred), torch.where(T > 0, ycf_pred, yf_pred)
        pehe_ts = torch.sqrt(loss((y1_pred - y0_pred), (Y1 - Y0)))
        ate = torch.abs(torch.mean((y1_pred - y0_pred)) - torch.mean((Y1 - Y0)))
    else:
        yf_pred,_ = model(X,T)
        ycf_pred,_ = model(X,1-T)
        y1_pred, y0_pred = torch.where(T > 0, yf_pred, ycf_pred), torch.where(T > 0, ycf_pred, yf_pred)
        pehe_ts = torch.sqrt(loss((y1_pred - y0_pred), (Y1 - Y0)))
        ate = torch.abs(torch.mean((y1_pred - y0_pred)) - torch.mean((Y1 - Y0)))
    return pehe_ts.item(), ate.item()




# X, T, Y1, Y0 = load_ihdp('./dataset/IHDP/ihdp_sample.csv')
if __name__ == "__main__":
    before_pehe = []
    before_ate = []
    after_pehe = []
    after_ate = []
    cfrnet_pehe = []
    cfrnet_ate = []
    all_pehe = []
    all_ate = []
    seed_torch = 21
    for i in range(10):
        np.random.seed(i)
        torch.manual_seed(seed_torch)
        params = {'lr': 0.0001, "weight_decay": 1e-4}
        X, T, Y1, Y0 = load_ihdp('./dataset/IHDP/ihdp_sample.csv')
        X_train_L, Y1_train_L, Y0_train_L, X_train_U, Y1_train_U, Y0_train_U, X_test, Y1_test, Y0_test, T_train_L, T_train_U, T_test = split_L_U(
            X, T, Y1, Y0)
        net = trinet.Tri_Net().cuda()
        net = train.train(net, X_train_L, Y1_train_L, Y0_train_L, T_train_L, epoch_num=500, batch_size=100, net_flag=0,
                          params=params)
        net = train.update(net, X_train_L, Y1_train_L, Y0_train_L, T_train_L, T_train_U, X_train_U, epoch_num=1,
                            params=params)
        pehe_after, ate_after = eval(net, X_test, T_test, Y1_test, Y0_test,update=True)
        after_pehe.append(pehe_after)
        after_ate.append(ate_after)
        print(pehe_after, ate_after)
        seed_torch = seed_torch + 1
    print("===============================")
    print("trinet pehe mean:"+str(sum(after_pehe)/len(after_pehe)))
    print("trinet ate mean:"+str(sum(after_ate)/len(after_ate)))
