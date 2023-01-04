import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
import time
import torchvision
import pdb
import sys
import copy
import pandas as pd
import logging
import torch.nn as nn
from choose import choose_data


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    norm = float(norm)
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def wasserstein(x, y, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]
    x = x.squeeze()
    y = y.squeeze()
    #    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x, y)  # distance_matrix(x,y,p=2)

    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, 10.0 / (nx * ny))
    delta = torch.max(M_drop).detach().cpu()
    eff_lam = (lam / M_mean).detach().cpu()

    '''compute new distance matrix'''
    Mt = M
    row = delta * torch.ones(M[0:1, :].shape)
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape), torch.zeros((1, 1))], 0)
    if cuda:
        row = row.cuda()
        col = col.cuda()
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    '''compute marginal'''
    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a

    u = a

    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.cuda()
    v = b / (torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.cuda()

    upper_t = u * (torch.t(v) * K).detach()

    E = upper_t * Mt
    D = 2 * torch.sum(E)

    if cuda:
        D = D.cuda()

    return D, Mlam


def train(net, X_train_L, Y1_train_L, Y0_train_L, T, epoch_num, batch_size, net_flag, params, beta=0.0001):
    weight_decay = params["weight_decay"]
    yf_train = torch.where(T > 0, Y1_train_L, Y0_train_L)
    if net_flag == 0:
        criterion = nn.MSELoss()
        optimizer = optim.Adam([{'params': net.parameters()}], lr=params['lr'], weight_decay=weight_decay)
        for epoch in range(epoch_num):
            net.train()
            optimizer.zero_grad()
            yf_pre_1, yf_pre_2, yf_pre_3, rep = net(X_train_L, T, net_flag)
            rep_t1 = rep[(T > 0).nonzero()]
            rep_t0 = rep[(T < 1).nonzero()]
            dist, _ = wasserstein(rep_t1, rep_t0, cuda=True)
            loss_train = criterion(yf_train, yf_pre_1) + criterion(yf_train, yf_pre_2) + criterion(yf_train,
                                                                                                   yf_pre_3) + beta * dist
            loss_train.backward()
            optimizer.step()


    if net_flag == 1:
        print("update net1")
        criterion = nn.MSELoss()
        '''
        optimizer = optim.SGD([{'params': net.Fshare.parameters()}, {'params': net.class1.parameters()}],
                               lr=params['lr'], momentum=0.9,weight_decay=weight_decay)
        '''
        optimizer = optim.SGD([{'params': net.Fshare.parameters()}, {'params': net.class1.parameters()}],
                              lr=params['lr'], momentum=0.9, weight_decay=weight_decay)
        for epoch in range(epoch_num):
            start_time = time.time()
            net.train()
            optimizer.zero_grad()
            yf_pre_1, rep = net(X_train_L, T, net_flag)
            rep_t1 = rep[(T > 0).nonzero()]
            rep_t0 = rep[(T < 1).nonzero()]
            dist, _ = wasserstein(rep_t1, rep_t0, cuda=True)
            loss_train = criterion(yf_train, yf_pre_1) + beta * dist
            # loss_train = torch.mean((yf_train-yf_pre_1)**2)
            loss_train.backward(retain_graph=True)
            optimizer.step()
            if epoch % 10 == 0:
                print("traning epoch " + str(epoch) + " loss:" + str(loss_train.item()) + " net1 loss:" + str(
                    criterion(yf_train, yf_pre_1).item()) + " time:", time.time() - start_time)

    if net_flag == 2:
        print("update net2")
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.class2.parameters(), lr=params['lr'], momentum=0.9,weight_decay=weight_decay)
        for epoch in range(epoch_num):
            net.train()
            optimizer.zero_grad()
            yf_pre_2, rep = net(X_train_L, T, net_flag)
            rep_t1 = rep[(T > 0).nonzero()]
            rep_t0 = rep[(T < 1).nonzero()]
            dist, _ = wasserstein(rep_t1, rep_t0, cuda=True)
            loss_train = criterion(yf_train, yf_pre_2) + beta * dist
            loss_train.backward(retain_graph=True)
            optimizer.step()
            if epoch % 10 == 0:
                print("traning epoch " + str(epoch) + " loss:" + str(loss_train.item()) + " net2 loss:",
                      criterion(yf_train, yf_pre_2).item())

    if net_flag == 3:
        print("update net3")
        criterion = nn.MSELoss()
        optimizer = optim.Adam([{'params': net.class3.parameters()}], lr=params['lr'], weight_decay=weight_decay)
        for epoch in range(epoch_num):
            net.train()
            optimizer.zero_grad()
            yf_pre_3, rep = net(X_train_L, T, net_flag)
            rep_t1 = rep[(T > 0).nonzero()]
            rep_t0 = rep[(T < 1).nonzero()]
            dist, _ = wasserstein(rep_t1, rep_t0, cuda=True)
            loss_train = criterion(yf_train, yf_pre_3) + beta * dist
            loss_train.backward(retain_graph=True)
            optimizer.step()
            if epoch % 10 == 0:
                print("traning epoch " + str(epoch) + " loss:" + str(loss_train.item()) + " net3 loss:",
                      criterion(yf_train, yf_pre_3).item())
    return net


def update(net, X_train_L, Y1_train_L, Y0_train_L, T_train_L, T_train_U, X_train_U, epoch_num, params, beta=0.0001):
    lr = params['lr']
    weight_decay = params['weight_decay']
    # round_ = len(X_train_U)
    round_ = 100
    add_idx = []
    for i in range(round_):
        print("starting choose data:", i)
        net1_X_train_add, net1_Y1_train_add, net1_Y0_train_add, net1_T_train_L_add, index11, index10 = choose_data(net,
                                                                                                                   X_train_U,
                                                                                                                   T_train_U,
                                                                                                                   X_train_L,
                                                                                                                   Y1_train_L,
                                                                                                                   Y0_train_L,
                                                                                                                   T_train_L,
                                                                                                                   1)

        net2_X_train_add, net2_Y1_train_add, net2_Y0_train_add, net2_T_train_L_add, index21, index20 = choose_data(net,
                                                                                                                   X_train_U,
                                                                                                                   T_train_U,
                                                                                                                   X_train_L,
                                                                                                                   Y1_train_L,
                                                                                                                   Y0_train_L,
                                                                                                                   T_train_L,
                                                                                                                   2)

        net3_X_train_add, net3_Y1_train_add, net3_Y0_train_add, net3_T_train_L_add, index31, index30 = choose_data(net,
                                                                                                                   X_train_U,
                                                                                                                   T_train_U,
                                                                                                                   X_train_L,
                                                                                                                   Y1_train_L,
                                                                                                                   Y0_train_L,
                                                                                                                   T_train_L,
                                                                                                                   3)
        add_idx.extend([int(index10), int(index11), int(index20), int(index30), int(index31)])

        for net_param in net.Fshare.parameters():
            net_param.requires_grad = True

        for net_param in net.class1.parameters():
            net_param.requires_grad = True
        net = train(net, net1_X_train_add, net1_Y1_train_add, net1_Y0_train_add, net1_T_train_L_add,
                      epoch_num=epoch_num, batch_size=10, net_flag=1, params=params, beta=beta)
        for net_param in net.Fshare.parameters():
            net_param.requires_grad = False
        for net_param in net.class1.parameters():
            net_param.requires_grad = False
        for net_param in net.class2.parameters():
            net_param.requires_grad = True
        net = train(net, net2_X_train_add, net2_Y1_train_add, net2_Y0_train_add, net2_T_train_L_add,
                    epoch_num=epoch_num, batch_size=10, net_flag=2, params=params,beta=beta)

        for net_param in net.class2.parameters():
            net_param.requires_grad = False
        for net_param in net.class3.parameters():
            net_param.requires_grad = True
        net = train(net, net3_X_train_add, net3_Y1_train_add, net3_Y0_train_add, net3_T_train_L_add,
                    epoch_num=epoch_num, batch_size=10, net_flag=3, params=params,beta=beta)
    return net
