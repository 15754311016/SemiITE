import torch
import torch.nn as nn
a=torch.Tensor([[1,2],[2,3]])
b=torch.Tensor([[1,2],[2,3]])
layer = nn.Linear(2,2)
output = layer(a)
loss1 = torch.nn.CrossEntropyLoss()
loss2 = torch.nn.MSELoss()
l = loss1(output,torch.Tensor([0,1]).long()) + loss2(output,b)
l.backward()