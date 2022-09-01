

from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, SRKMIX, FlashVLN, PropertyCorrelationsPackage, \
    HeatCapacityGas
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
from data import generate_data
from model import ArtificialNN
import torch
import torch.nn as nn
import thermo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import convolve2d, correlate2d
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
import torch
from mpi4py import MPI
import random
import os
from itertools import combinations
import csv
from pytorch_tabnet.tab_model import TabNetRegressor
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
All_ID = ['Methane','Ethane', 'Propane', 'N-Butane','N-Pentane', 'N-Hexane', 'Heptane']
id_index=0
IDs=list(combinations(All_ID,2))
IDs=['Ethane','N-Pentane']
constants, properties = ChemicalConstantsPackage.from_IDs(IDs)
dic={"material":All_ID,"Tc":constants.Tcs,"Pc":constants.Pcs,"Ac":constants.omegas}
print(dic)

num=100
Ts=[310]*num
Ps=np.linspace(1e5,10e5,num)
Zs=[[0.5,0.5]]*num

# Ts=np.linspace(1e1,1e3,num)
# Ps=[1e5]*num
# Zs=[[0.5,0.5]]*num
print(len(Ts))
flash=generate_data.flashdata(constants, properties, {"T":Ts,"P":Ps},Zs,"Vapor_Liquid")
data_loader = DataLoader(flash, shuffle=False,batch_size=1,collate_fn=generate_data.collector(return_type="NParray"))

#ANN
ANNkwargs={}
ANNkwargs["Nodes_per_layer"] = 500
ANNkwargs["deepth"] = 5
ANNkwargs["material"] = IDs
model_ANN=ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.simple_ANN, ANNkwargs)
model_ANN.load_model(
    "E:\\Ic document\\IRP_flash_calculation_accelaration\\Simple_ANN_experience\\simpleANN_500.pth")

#XGB

model_XGB= XGBRegressor()
model_XGB.load_model(
    "E:\\Ic document\\IRP_flash_calculation_accelaration\\XGB_experience\\mixture.json")

mean=np.array([4.29042626e+06, 3.32385527e+06, 3.27222181e+02, 4.75159252e+02,
 1.20214372e-01, 2.64128853e-01, 5.05477944e+02, 5.50226799e+05,
 4.99803760e-01, 5.00196240e-01])
std=np.sqrt(np.array([2.99937521e+11, 3.46382041e+11, 1.06477049e+04, 4.25280160e+03,
 7.79023299e-03, 5.43348940e-03, 8.16438270e+04, 6.75487152e+10,
 8.30771571e-02, 8.30771571e-02]))

print(model_ANN.get_device)
instance_id=0
Ethe_Pen_g = {}
Ethe_Pen_g["T"]=[]
Ethe_Pen_g["P"] = []
Ethe_Pen_g["v_1"] = []
Ethe_Pen_g["v_2"] = []
Ethe_Pen_g["l_1"] = []
Ethe_Pen_g["l_2"] = []
Ethe_Pen_g["V"] = []
Ethe_Pen_g["L"] = []


Ethe_Pen_ANN = {}
Ethe_Pen_ANN["T"]=[]
Ethe_Pen_ANN["P"] = []
Ethe_Pen_ANN["v_1"] = []
Ethe_Pen_ANN["v_2"] = []
Ethe_Pen_ANN["l_1"] = []
Ethe_Pen_ANN["l_2"] = []
Ethe_Pen_ANN["V"] = []
Ethe_Pen_ANN["L"] = []


Ethe_Pen_XGB = {}
Ethe_Pen_XGB["T"]=[]
Ethe_Pen_XGB["P"] = []
Ethe_Pen_XGB["v_1"] = []
Ethe_Pen_XGB["v_2"] = []
Ethe_Pen_XGB["l_1"] = []
Ethe_Pen_XGB["l_2"] = []
Ethe_Pen_XGB["V"] = []
Ethe_Pen_XGB["L"] = []

for i,(x,y) in enumerate(data_loader):

    input_data=np.array([np.concatenate([constants.Pcs,constants.Tcs,constants.omegas,[Ts[instance_id]],[Ps[instance_id]],Zs[instance_id]],axis=0)])

    Ethe_Pen_g["T"].append(Ts[i])
    Ethe_Pen_g["P"].append(Ps[i])
    Ethe_Pen_g["v_1"].append(y[0][0])
    Ethe_Pen_g["v_2"].append(y[0][1])
    Ethe_Pen_g["l_1"].append(y[0][2])
    Ethe_Pen_g["l_2"].append(y[0][3])
    Ethe_Pen_g["V"].append(y[0][4])
    Ethe_Pen_g["L"].append(y[0][5])


    predANN=model_ANN.predict((x - mean) / std)


    Ethe_Pen_ANN["T"].append(Ts[i])
    Ethe_Pen_ANN["P"].append(Ps[i])
    Ethe_Pen_ANN["v_1"].append(predANN[0][0])
    Ethe_Pen_ANN["v_2"].append(predANN[0][1])
    Ethe_Pen_ANN["l_1"].append(predANN[0][2])
    Ethe_Pen_ANN["l_2"].append(predANN[0][3])
    Ethe_Pen_ANN["V"].append(predANN[0][4])
    Ethe_Pen_ANN["L"].append(predANN[0][5])

    predXGB=model_XGB.predict((x - mean) / std)


    Ethe_Pen_XGB["T"].append(Ts[i])
    Ethe_Pen_XGB["P"].append(Ps[i])
    Ethe_Pen_XGB["v_1"].append(predXGB[0][0])
    Ethe_Pen_XGB["v_2"].append(predXGB[0][1])
    Ethe_Pen_XGB["l_1"].append(predXGB[0][2])
    Ethe_Pen_XGB["l_2"].append(predXGB[0][3])
    Ethe_Pen_XGB["V"].append(predXGB[0][4])
    Ethe_Pen_XGB["L"].append(predXGB[0][5])





data_g=pd.DataFrame(Ethe_Pen_g)
data_ANN=pd.DataFrame(Ethe_Pen_ANN)
data_XGB=pd.DataFrame(Ethe_Pen_XGB)
fig=plt.figure(figsize=(16, 4))


sparse=20
x="P"
comment="(K)"
#1
ax1 = fig.add_subplot(1, 6, 1)
ax1.plot(data_g[x],data_g['v_1'],color='black',label="ground_truth")
ax1.scatter(data_ANN[x][::int(num / sparse)], data_ANN['v_1'][::int(num / sparse)], marker='o', s=10,label="ANN")
ax1.scatter(data_XGB[x][::int(num / sparse)], data_XGB['v_1'][::int(num / sparse)], marker='o', s=10,label="XGB")
ax1.set_title('v_1')
ax1.set_xlabel(x+comment)
ax1.legend()

#1
ax2 = fig.add_subplot(1, 6, 2)
ax2.plot(data_g[x],data_g['v_2'],color='black',label="ground_truth")
ax2.scatter(data_ANN[x][::int(num / sparse)], data_ANN['v_2'][::int(num / sparse)], marker='o', s=10,label="ANN")
ax2.scatter(data_XGB[x][::int(num / sparse)], data_XGB['v_2'][::int(num / sparse)], marker='o', s=10,label="XGB")
ax2.set_title('v_2')
ax2.set_xlabel(x+comment)
ax2.legend()
#1
ax3 = fig.add_subplot(1, 6, 3)
ax3.plot(data_g[x],data_g['l_1'],color='black',label="ground_truth")
ax3.scatter(data_ANN[x][::int(num / sparse)], data_ANN['l_1'][::int(num / sparse)], marker='o', s=10,label="ANN")
ax3.scatter(data_XGB[x][::int(num / sparse)], data_XGB['l_1'][::int(num / sparse)], marker='o', s=10,label="XGB")
ax3.set_title('l_1')
ax3.set_xlabel(x+comment)
ax3.legend()
#1
ax4 = fig.add_subplot(1, 6, 4)
ax4.plot(data_g[x],data_g['l_2'],color='black',label="ground_truth")
ax4.scatter(data_ANN[x][::int(num / sparse)], data_ANN['l_2'][::int(num / sparse)], marker='o', s=10,label="ANN")
ax4.scatter(data_XGB[x][::int(num / sparse)], data_XGB['l_2'][::int(num / sparse)], marker='o', s=10,label="XGB")
ax4.set_title('l_2')
ax4.set_xlabel(x+comment)
ax4.legend()
#1
ax5 = fig.add_subplot(1, 6, 5)
ax5.plot(data_g[x],data_g['V'],color='black',label="ground_truth")
ax5.scatter(data_ANN[x][::int(num / sparse)], data_ANN['V'][::int(num / sparse)], marker='o', s=10,label="ANN")
ax5.scatter(data_XGB[x][::int(num / sparse)], data_XGB['V'][::int(num / sparse)], marker='o', s=10,label="XGB")
ax5.set_title('V')
ax5.set_xlabel(x+comment)
ax5.legend()
#1
ax6 = fig.add_subplot(1, 6, 6)
ax6.plot(data_g[x],data_g['L'],color='black',label="ground_truth")
ax6.scatter(data_ANN[x][::int(num / sparse)], data_ANN['L'][::int(num / sparse)], marker='o', s=10,label="ANN")
ax6.scatter(data_XGB[x][::int(num / sparse)], data_XGB['L'][::int(num / sparse)], marker='o', s=10,label="XGB")
ax6.set_title('L')
ax6.set_xlabel(x+comment)
ax6.legend()

ax1.set_ylabel(IDs[0]+"-"+IDs[1])
plt.show()

    # print("x: ",x)
    # print("model:",tabnet.predict(x))
    # print("ground Truth:",y)




