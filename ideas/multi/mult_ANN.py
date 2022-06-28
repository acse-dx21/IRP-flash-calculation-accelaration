import torch

import torch.nn as nn
import sys
sys.path.append("..")
from tool.log import epoch_log

class mult_ANN(nn.Module):
    def __init__(self):
        super(mult_ANN, self).__init__()
        self.l1=nn.Linear(3,100)
        self.l2 = nn.Linear(self.l1.out_features, 200)

        self.output = nn.Linear(self.l2.out_features,1)
        self.act=nn.ELU()
        self.material_num=3

    def forward(self,x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x =self.output(x)
        return x


criterian =nn.MSELoss()
multi=mult_ANN()
opti=torch.optim.Adam(multi.parameters(),lr=0.001)
logs=epoch_log("multi")

for i in range(100000):
    x=torch.randint(-10,10,(20,2)).float()

    opti.zero_grad()
    y=x[:,0]*x[:,1]
    x = torch.concat([x, y.unsqueeze(1)],axis=1)

    ypred=multi(x)
    loss=criterian(ypred,y)
    print(loss)
    loss.backward()

    opti.step()
    logs({"epoch": i, "loss": loss.item()})
logs.release()
import pandas as pd
import seaborn as sns
data=pd.read_csv("multi.csv")
import matplotlib.pyplot as plt
plt.plot(data["epoch"].tolist(),data["loss"].tolist())
plt.show()
