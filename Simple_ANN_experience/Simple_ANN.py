import sys

sys.path.append("..")
from data import generate_data
import tool
import xgboost as xgb
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



xgbReg = xgb.XGBRegressor(n_estimators=200)
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
from model.train import check_IDexist
from tool.log import log

import os

data_path = ".." + os.sep + "cleaned_data" + os.sep
result_path = "." + os.sep + "MSELoss_data" + os.sep

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
    rank = comm.Get_rank()
    size = comm.Get_size()

    all_data = generate_data.multicsv_data_generater(data_path, return_type="Dataloader")

    for i in range(len(all_data) - rank - 1, -1, -size):
        Material_ID=all_data.materials[i]
        print(Material_ID, check_IDexist(Material_ID, result_path))

        if not check_IDexist(Material_ID, result_path):
            batchsize_list = [64, 128, 512]
            learning_rate_list = [0.001, 0.002, 0.005]
            for batchsize in batchsize_list:
                for learning_rate in learning_rate_list:

                    all_data.batch_size=batchsize
                    train_loader, test_loader, Material_ID = all_data[i]

                    my_model = ArtificialNN.simple_ANN(Material_ID)
                    #         model.load_state_dict(torch.load(model_path))

                    criterion = nn.MSELoss()
                    #         criterion=loss_function.My_Mass_Balance_loss(1,1)
                    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)


                    ANN_train = DeepNetwork_Train(my_model, criterion, optimizer, train_loader,
                                                  "." + os.sep + f"{type(criterion).__name__}_data" + os.sep + "mix_" + str(
                                                      len(
                                                          Material_ID)) + os.sep + str(Material_ID),test_loader,f"lr={learning_rate}bs={batchsize}")

                    ANN_train.Train(epoch=1)

                    model_paths = "." + os.sep + "saved_model" + os.sep + "simpleANN" + f"lr={learning_rate}bs={batchsize}" + type(
                        criterion).__name__ + ".pt"
                    torch.save(my_model.state_dict(), model_paths)

