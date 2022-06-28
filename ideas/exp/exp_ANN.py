import torch

import torch.nn as nn
import sys
sys.path.append("..\\..")
from tool.log import epoch_log

class exp_ANN(nn.Module):
    def __init__(self):
        super(exp_ANN, self).__init__()
        self.l1=nn.Linear(1,100)
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
exp=exp_ANN()
opti=torch.optim.Adam(exp.parameters(), lr=0.01)
logs=epoch_log("exp")

for i in range(50000):
    x=torch.randint(-1,10,(20,1)).float()

    opti.zero_grad()
    y=torch.exp((x))
    ypred=exp(x)
    loss=criterian(ypred,y)

    loss.backward()

    opti.step()
    logs({"epoch": i, "loss": loss.item()})
logs.release()
import pandas as pd
import seaborn as sns
data=pd.read_csv("exp.csv")
import matplotlib.pyplot as plt
plt.plot(data["epoch"].tolist(),data["loss"].tolist())
plt.show()


import numpy as np
test=torch.tensor(np.linspace(-1,10,1000),dtype=torch.float32).unsqueeze(1)
print()


# data=pd.read_csv("reci.csv")
import matplotlib.pyplot as plt
plt.plot(test.tolist(),exp(test).tolist(),"r--",label="pred")
plt.plot(test.tolist(),torch.exp(test).tolist(),"b--",label="real")
plt.legend()
plt.show()
