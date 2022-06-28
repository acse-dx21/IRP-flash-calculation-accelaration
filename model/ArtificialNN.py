import torch

import torch.nn as nn
from torch.autograd.function import Function as Function
class simple_ANN(nn.Module):
    def __init__(self,material):
        super(simple_ANN,self).__init__()
        self.material=material
        self.material_num=len(material)
        self.l1=nn.Linear(self.material_num*4+2,50)
        self.l2 = nn.Linear(self.l1.out_features, 100)
        self.l3 = nn.Linear(self.l2.out_features, 30)
        self.output = nn.Linear(self.l3.out_features, self.material_num*2+2)
        self.act=nn.Sigmoid()


    def forward(self,x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        x = self.act(self.output(x))
        return x


class deeper_ANN(nn.Module):
    def __init__(self):
        super(deeper_ANN,self).__init__()
        self.l1=nn.Linear(14,50)
        self.l2 = nn.Linear(self.l1.out_features, 100)
        self.l3 = nn.Linear(self.l2.out_features, 200)

        self.l4 = nn.Linear(self.l3.out_features, 100)
        self.l5 = nn.Linear(self.l4.out_features, 50)

        self.output = nn.Linear(self.l5.out_features, 8)
        self.act=nn.Sigmoid()
        self.material_num=3

    def forward(self,x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        x = self.act(self.l4(x))
        x = self.act(self.l5(x))
        x = self.act(self.output(x))
        return x



class My_MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))





class Mass_Balence_loss(nn.Module):
    def __init__(self,Beta=None):
        self.Beta=Beta if Beta is not None else 1

    def forward(self,M1,M2):
        return torch.abs(M1-M2)