import generate_data
from data.generate_data import genertate_T, genertate_P, genertate_Zs_n,flashdata
# from tool import Py2Cpp as pc
from my_test import test_data_set
import xgboost as xgb
from xgboost import XGBRegressor,XGBClassifier
from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, SRKMIX, FlashVLN, PropertyCorrelationsPackage, \
    HeatCapacityGas
from torch.utils.data import DataLoader
from model import ArtificialNN
from model.train import DeepNetwork_Train ,check_IDexist
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

import csv


#
All_ID = ['Methane','Ethane', 'Propane', 'N-Butane','N-Pentane', 'N-Hexane', 'Heptane']
print("All_IDs",len(All_ID))



from itertools import combinations
# # from multiprocessing import Pool
def f(ID):
    i=len(ID)
    # print((ID))
    num=3
    root_path = "."+os.sep+"mini_cleaned_data"+os.sep +f"mini_data_{num}" + os.sep + "mix_"+str(i)+ os.sep

    constants, properties = ChemicalConstantsPackage.from_IDs(ID)
    # generate_data.generate_good_TPZ(1000,constants, properties,root_path+str(ID)+"_test",comment=str(ID))
    data=pd.read_csv(root_path + str(ID) + "_train.csv",comment="#",index_col=[0])
    if(len(data)<11500):
        generate_data.generate_good_TPZ(2000, constants, properties, root_path + str(ID) + "_train", comment=str(ID))
    else:
        data=data.iloc[:12000]
        data.to_csv(root_path + str(ID) + "_train.csv")

    f=open(root_path + str(ID) + "_train.csv",mode="a+")

    f.write("# "+str(ID)+os.sep+"n")

    print(data.shape)

    return 0

IDs=[]
if __name__ == '__main__':
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()

    size=comm.Get_size()
    # if True:
    #     for com in combinations(All_ID,2):
    #         IDs.append(com)
    for i in range(7):
        for com in combinations(All_ID,i+1):
            IDs.append(com)
    for i in range(rank,len(IDs),size):
        f(IDs[i])



