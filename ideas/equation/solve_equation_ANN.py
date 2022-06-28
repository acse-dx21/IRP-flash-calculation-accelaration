import torch

import torch.nn as nn
import sys
sys.path.append("..")
from tool.log import epoch_log

class equation_ANN(nn.Module):
    def __init__(self):
        super(equation_ANN, self).__init__()
        self.l1=nn.Linear(4,50)
        self.l2 = nn.Linear(self.l1.out_features, 100)

        self.output = nn.Linear(self.l2.out_features,2)
        self.act=nn.PReLU()
        self.material_num=3

    def forward(self,x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x =self.output(x)
        return x


criterian =nn.MSELoss()
equation=equation_ANN()
opti=torch.optim.Adam(equation.parameters(),lr=0.001)
logs=epoch_log("equation")
args=[1, -2, 0]
import numpy as np


def check(x):
    if(len(x)==2):

        if np.issubdtype(type(x[0]), np.float64) or np.issubdtype(type(x[0]), np.float32):

            return True
        else: return False
    else:return False


# for i in range(100000):
#     x=torch.randint(-10,10,(1,3)).float()
#
#     root=np.roots(x[0])
#
#     if check(root):
#
#         root=torch.tensor(root,dtype=torch.float32)
#         opti.zero_grad()
#
#
#         x_4=torch.concat((x,torch.tensor([[x[0,0].item()*x[0,2].item()]])),dim=1)
#
#
#         ypred=equation(x_4)
#         loss=criterian(ypred,root)
#         print(loss)
#         loss.backward()
#
#         opti.step()
#         logs({"epoch": i, "loss": loss.item()})
# logs.release()
import pandas as pd
import seaborn as sns
data=pd.read_csv("equation.csv")
import matplotlib.pyplot as plt
plt.scatter(data["epoch"].tolist(),data["loss"].tolist())
plt.show()
