import torch
import torch.nn as nn
import numpy as np


def choose_data(net, X_train_U, T_train_U, X_train_L, Y1_train_L, Y0_train_L, T_train_L, net_flag):
    # get all the outputs from net1 2 3
    #print("before add:",X_train_L.shape)
    pack = 1
    output1_f, output2_f, output3_f, _ = net(X_train_U, T_train_U, 0)
    output1_cf, output2_cf, output3_cf, _ = net(X_train_U, 1 - T_train_U, 0)
    # get y_1 and y_0 from net1
    output1_t1 = torch.where(T_train_U > 0, output1_f, output1_cf)
    output1_t0 = torch.where(T_train_U > 0, output1_cf, output1_f)
    # get y_1 and y_0 from net2
    output2_t1 = torch.where(T_train_U > 0, output2_f, output2_cf)
    output2_t0 = torch.where(T_train_U > 0, output2_cf, output2_f)
    # get y_1 and y_0 from net3
    output3_t1 = torch.where(T_train_U > 0, output3_f, output3_cf)
    output3_t0 = torch.where(T_train_U > 0, output3_cf, output3_f)
    if net_flag == 1:
        diff_abs_t1 = torch.abs(output2_t1 - output3_t1)
        sort_index_t1 = torch.argsort(diff_abs_t1)
        diff_abs_t0 = torch.abs(output2_t0 - output3_t0)
        sort_index_t0 = torch.argsort(diff_abs_t0)
        X_selected_t1 = X_train_U[sort_index_t1[:pack]]
        if np.random.randint(0, 2) == 0:
            y_selected_t1 = output2_t1[sort_index_t1[:pack]]
        else:
            y_selected_t1 = output3_t1[sort_index_t1[:pack]]

        X_selected_t0 = X_train_U[sort_index_t0[:pack]]
        if np.random.randint(0, 2) == 0:
            y_selected_t0 = output2_t0[sort_index_t0[:pack]]
        else:
            y_selected_t0 = output3_t0[sort_index_t0[:pack]]
        #print("Y1:",Y1_train_L)
        #print("Y_selected",y_selected_t1)
    if net_flag == 2:
        diff_abs_t1 = torch.abs(output1_t1 - output3_t1)
        sort_index_t1 = torch.argsort(diff_abs_t1)
        diff_abs_t0 = torch.abs(output1_t0 - output3_t0)
        sort_index_t0 = torch.argsort(diff_abs_t0)

        X_selected_t1 = X_train_U[sort_index_t1[:pack]]
        if np.random.randint(0, 2) == 0:
            y_selected_t1 = output1_t1[sort_index_t1[:pack]]
        else:
            y_selected_t1 = output3_t1[sort_index_t1[:pack]]

        X_selected_t0 = X_train_U[sort_index_t0[:pack]]
        if np.random.randint(0, 2) == 0:
            y_selected_t0 = output1_t0[sort_index_t0[:pack]]
        else:
            y_selected_t0 = output3_t0[sort_index_t0[:pack]]

    if net_flag == 3:
        diff_abs_t1 = torch.abs(output2_t1 - output1_t1)
        sort_index_t1 = torch.argsort(diff_abs_t1)
        diff_abs_t0 = torch.abs(output2_t0 - output1_t0)
        sort_index_t0 = torch.argsort(diff_abs_t0)

        X_selected_t1 = X_train_U[sort_index_t1[:pack]]
        if np.random.randint(0, 2) == 0:
            y_selected_t1 = output2_t1[sort_index_t1[:pack]]
        else:
            y_selected_t1 = output1_t1[sort_index_t1[:pack]]

        X_selected_t0 = X_train_U[sort_index_t0[:pack]]
        if np.random.randint(0, 2) == 0:
            y_selected_t0 = output2_t0[sort_index_t0[:pack]]
        else:
            y_selected_t0 = output1_t0[sort_index_t0[:pack]]
    X_train_add = torch.cat((X_train_L, X_selected_t1, X_selected_t0), dim=0)
    Y1_train_add = torch.cat((Y1_train_L, y_selected_t1, torch.FloatTensor([0]).cuda()))
    Y0_train_add = torch.cat((Y0_train_L, y_selected_t0, torch.FloatTensor([0]).cuda()))
    T_train_L_add = torch.cat((T_train_L, torch.LongTensor([1]).cuda(), torch.LongTensor([0]).cuda()))
    #print("after add:",X_train_add.shape)

    return X_train_add.detach(), Y1_train_add.detach(), Y0_train_add.detach(), T_train_L_add.detach(), sort_index_t1[:pack].detach(), sort_index_t0[:pack].detach()
