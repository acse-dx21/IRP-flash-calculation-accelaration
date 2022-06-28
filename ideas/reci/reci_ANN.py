import torch

import torch.nn as nn
import sys
sys.path.append("..")
from tool.log import epoch_log

class reci_ANN(nn.Module):
    def __init__(self):
        super(reci_ANN, self).__init__()
        self.l1=nn.Linear(1,100)
        self.l2 = nn.Linear(self.l1.out_features, 200)

        self.output = nn.Linear(self.l2.out_features,1)
        self.act=nn.PReLU()
        self.material_num=3

    def forward(self,x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x =self.output(x)
        return x


criterian =nn.MSELoss()
reci=reci_ANN()
opti=torch.optim.Adam(reci.parameters(),lr=0.001)
logs=epoch_log("reci")

for i in range(20000):
    x=torch.randint(1,500,(20,1)).float()

    opti.zero_grad()
    y=1/(x)

    ypred=reci(x)
    loss=criterian(ypred,y)
    print(loss)
    loss.backward()

    opti.step()
    logs({"epoch": i, "loss": loss.item()})
logs.release()
import pandas as pd
import numpy as np
test=torch.tensor(np.linspace(1,100,100),dtype=torch.float32).unsqueeze(1)
print()


# data=pd.read_csv("reci.csv")
import matplotlib.pyplot as plt
plt.plot(test.tolist(),reci(test).tolist(),"r--",label="pred")

plt.plot(test.tolist(),(1/(test)).tolist(),"b--",label="real")
plt.legend()
plt.show()
