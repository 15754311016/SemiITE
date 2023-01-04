import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init
import pdb
import torchvision.transforms as transforms
import numpy as np


class Net_share(nn.Module):
    def __init__(self, fea_dim, hid_dim, n_in, dropout):
        super(Net_share, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(fea_dim, hid_dim).cuda()])
        for i in range(n_in - 1):
            self.fc.append(nn.Linear(hid_dim, hid_dim).cuda())
        self.dropout = dropout

    def forward(self, x):
        rep = F.relu(self.fc[0](x))
        rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, len(self.fc) - 1):
            rep = F.relu(self.fc[i](rep))
            rep = F.dropout(rep, self.dropout, training=self.training)
        return rep


class Net1(nn.Module):
    def __init__(self, hid_dim, dropout):
        super(Net1, self).__init__()
        self.layer1_t0 = nn.Linear(hid_dim, 200).cuda()
        self.layer2_t0 = nn.Linear(200, 300).cuda()
        self.layer3_t0 = nn.Linear(300, 1).cuda()

        self.layer1_t1 = nn.Linear(hid_dim, 200).cuda()
        self.layer2_t1 = nn.Linear(200, 300).cuda()
        self.layer3_t1 = nn.Linear(300, 1).cuda()
        self.dropout = dropout

    def forward(self, x, t):
        rep_t0 = F.elu(self.layer1_t0(x))
        rep_t0 = F.dropout(rep_t0, self.dropout, training=self.training)
        rep_t0 = F.elu(self.layer2_t0(rep_t0))
        rep_t0 = F.dropout(rep_t0, self.dropout, training=self.training)
        y_0 = self.layer3_t0(rep_t0)

        rep_t1 = F.elu(self.layer1_t1(x))
        rep_t1 = F.dropout(rep_t1, self.dropout, training=self.training)
        rep_t1 = F.elu(self.layer2_t1(rep_t1))
        rep_t1 = F.dropout(rep_t1, self.dropout, training=self.training)
        y_1 = self.layer3_t1(rep_t1)
        y_0 = y_0.view(-1)
        y_1 = y_1.view(-1)
        y = torch.where(t > 0, y_1, y_0)
        return y


class Net2(nn.Module):
    def __init__(self, hid_dim,  dropout):
        super(Net2, self).__init__()
        self.layer1_t0 = nn.Linear(hid_dim, 300).cuda()
        self.layer2_t0 = nn.Linear(300, 500).cuda()
        self.layer3_t0 = nn.Linear(500, 1).cuda()

        self.layer1_t1 = nn.Linear(hid_dim, 300).cuda()
        self.layer2_t1 = nn.Linear(300, 500).cuda()
        self.layer3_t1 = nn.Linear(500, 1).cuda()
        self.dropout = dropout

    def forward(self, x, t):
        rep_t0 = F.relu(self.layer1_t0(x))
        rep_t0 = F.dropout(rep_t0, self.dropout, training=self.training)
        rep_t0 = F.relu(self.layer2_t0(rep_t0))
        rep_t0 = F.dropout(rep_t0, self.dropout, training=self.training)
        y_0 = self.layer3_t0(rep_t0)

        rep_t1 = F.relu(self.layer1_t1(x))
        rep_t1 = F.dropout(rep_t1, self.dropout, training=self.training)
        rep_t1 = F.relu(self.layer2_t1(rep_t1))
        rep_t1 = F.dropout(rep_t1, self.dropout, training=self.training)
        y_1 = self.layer3_t1(rep_t1)
        y_0 = y_0.view(-1)
        y_1 = y_1.view(-1)
        y = torch.where(t > 0, y_1, y_0)
        return y


class Net3(nn.Module):
    def __init__(self, hid_dim, dropout):
        super(Net3, self).__init__()
        self.layer1_t0 = nn.Linear(hid_dim, 200).cuda()
        self.layer2_t0 = nn.Linear(200, 200).cuda()
        self.layer3_t0 = nn.Linear(200, 100).cuda()
        self.layer4_t0 = nn.Linear(100, 1).cuda()

        self.layer1_t1 = nn.Linear(hid_dim, 200).cuda()
        self.layer2_t1 = nn.Linear(200, 200).cuda()
        self.layer3_t1 = nn.Linear(200, 100).cuda()
        self.layer4_t1 = nn.Linear(100, 1).cuda()
        self.dropout = dropout

    def forward(self, x, t):
        rep_t0 = F.leaky_relu(self.layer1_t0(x))
        rep_t0 = F.dropout(rep_t0, self.dropout, training=self.training)
        rep_t0 = F.leaky_relu(self.layer2_t0(rep_t0))
        rep_t0 = F.dropout(rep_t0, self.dropout, training=self.training)
        rep_t0 = F.leaky_relu(self.layer3_t0(rep_t0))
        rep_t0 = F.dropout(rep_t0, self.dropout, training=self.training)
        y_0 = self.layer4_t0(rep_t0)

        rep_t1 = F.leaky_relu(self.layer1_t1(x))
        rep_t1 = F.dropout(rep_t1, self.dropout, training=self.training)
        rep_t1 = F.leaky_relu(self.layer2_t1(rep_t1))
        rep_t1 = F.dropout(rep_t1, self.dropout, training=self.training)
        rep_t1 = F.leaky_relu(self.layer3_t1(rep_t1))
        rep_t1 = F.dropout(rep_t1, self.dropout, training=self.training)
        y_1 = self.layer4_t1(rep_t1)
        y_0 = y_0.view(-1)
        y_1 = y_1.view(-1)
        y = torch.where(t > 0, y_1, y_0)
        return y


class Tri_Net(nn.Module):
    def __init__(self):
        super(Tri_Net, self).__init__()
        self.Fshare = Net_share(10,100,3,0.1)
        self.class1 = Net1(100,0.1)
        self.class2 = Net2(100,0.1)
        self.class3 = Net3(100,0.1)

    def forward(self, x, t, flag):
        x = self.Fshare(x)  # 128*16*16
        x1 = x
        x2 = x
        x3 = x
        # pdb.set_trace()
        if flag == 0:
            return self.class1(x1, t), self.class2(x2, t), self.class3(x3, t)
        elif flag == 1:
            return self.class1(x1, t)
        elif flag == 2:
            return self.class2(x2, t)
        elif flag == 3:
            return self.class3(x3, t)


if __name__ == "__main__":
    x = torch.randn(5, 10).cuda()
    Net = Tri_Net().cuda()
    t = torch.tensor([1, 1, 0, 0, 1]).cuda()
    flag=0
    y1, y2, y3 = Net(x,t,flag)
    for name, p in Net.named_parameters():
        print(name)
