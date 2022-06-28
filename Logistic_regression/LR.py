import sys

sys.path.append("..")
from data import generate_data
import tool
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from data.generate_data import genertate_T, genertate_P, genertate_Zs_n

from my_test import test_data_set
from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, SRKMIX, FlashVLN, PropertyCorrelationsPackage, \
    HeatCapacityGas
from torch.utils.data import DataLoader
from model import ArtificialNN
from model.train import DeepNetwork_Train
import torch
import torch.nn as nn
import thermo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputRegressor

import random

from sklearn.model_selection import GridSearchCV

All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
from itertools import combinations

from sklearn.model_selection import GridSearchCV
import time

param_grid = [
    # try combinations of hyperparameters
    {'subsample': [0.2, 0.6, 1.0],
     'learning_rate': [0.01, 0.05, 0.1],
     'n_estimators': [300, 400, 500],
     'max_depth': [3, 5, 10],
     'colsample_bytree': [0.6],
     'reg_lambda': [10]}
]

logis_Reg = MultiOutputRegressor(LogisticRegression(random_state=123))
def grid_i(X_train, y_train):
    # train across 3 folds
    # grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', n_jobs=3, random_state=42),
    #                            param_grid,
    #                            cv=3,
    #                            scoring='neg_mean_squared_error',
    #                            return_train_score=True,
    #                            verbose=1,
    #                            n_jobs=2)

    start = time.time()
    log_Reg.fit(X_train, y_train)
    print("Run time = ", time.time() - start)
    return log_Reg



# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
from model.train import check_IDexist
from tool.log import log

import os

data_path = ".." + os.sep + "cleaned_data" + os.sep
result_path = "." + os.sep + "result_data" + os.sep



# print(len(all_data))
# print(all_data[30][4])

# trainlabel = all_data.labels_train
# testlabel = all_data.labels_test
#
# for i in range(len(all_data)):
#
#     pass
from mpi4py import MPI

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank=comm.Get_rank()
    size=comm.Get_size()

    all_data = generate_data.multicsv_data_generater(data_path)

    for i in range(len(all_data)-rank-1,-1,-size):


        X_train, y_train, X_test, y_test, Material_ID = all_data[i]

        print(Material_ID, check_IDexist(Material_ID, result_path))

        if not check_IDexist(Material_ID, result_path):


            start = time.time()
            logis_Reg.fit(X_train, (160*y_train).astype('int'))
            print("Run time = ", time.time() - start)

            print("score: ",logis_Reg.score(X_test,(160*y_test).astype('int')))
