import sys

sys.path.append("..")
import os
from data import generate_data
import tool
import xgboost as xgb
from sklearn.metrics import mean_squared_error
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


from itertools import combinations

from sklearn.model_selection import GridSearchCV
import time

mini_data_path=".."+os.sep+"data"+os.sep+"mini_cleaned_data"+os.sep

All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
relate_data=generate_data.multicsv_data_generater(mini_data_path)

relate_data.set_batch_size(128)
X_train=0
y_train=0
X_test=0
y_test=0
Material_ID = 0

param_grid = [
    {'subsample': [0.2, 0.6, 1.0],
     'learning_rate': [0.01, 0.05, 0.1],
     'n_estimators': [300, 400, 500],
     'max_depth': [3, 5, 10],
     'colsample_bytree': [0.6],
     'reg_lambda': [10]}
]




from model.train import check_IDexist
from tool.log import log

import os

data_path = ".." + os.sep +"data"+os.sep+"mini_cleaned_data" + os.sep
result_path = "." + os.sep + "MSELoss_data" + os.sep


from mpi4py import MPI



# model=ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.simple_ANN,{"material":Material_ID})
# model.fit(Data_loader=train_loader,epoch=10)
# print(model.get_data)
# model.score(test_loader)
# print(model.get_data)

def get_related_path(Material_ID):
    return "mix_"+str(len(Material_ID))+os.sep+str(Material_ID)+".csv"



from bayes_opt import BayesianOptimization
data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}
def model_cv(**kwargs):
    kwargs["Nodes_per_layer"] = int(kwargs["Nodes_per_layer"])
    kwargs["deepth"] = int(kwargs["deepth"])
    kwargs["material"]=Material_ID
    model_instance = ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.simple_ANN,kwargs)

    # record_data
    start_train = time.time()
    model_instance.fit(X_train,y_train,epoch=30)
    score=model_instance.score(X_test,y_test)

    train_time = time.time() - start_train
    start_pred = time.time()
    pred = model_instance.predict(X_test)
    test_time = time.time() - start_pred

    data_record["trainning_time_consume(s)"].append(train_time)
    data_record["test_time_consume(s)"].append(test_time)

    epoch_root="."+os.sep+"BO_epoch_routing"+os.sep
    pd.DataFrame(model_instance.data_record).to_csv(epoch_root + get_related_path(Material_ID))

    return -score

import argparse
# print(a)
from mpi4py import MPI



def run_bayes_optimize(num_of_iteration=1,data_index=10):
    BO_root="."+os.sep+"BO_result_data"+os.sep
    BO_routing = "." + os.sep + "BO_training_routing" + os.sep
    global X_train, y_train, X_test, y_test, Material_ID

    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]

    rf_bo = BayesianOptimization(
            model_cv,
        {'Nodes_per_layer': [100, 1000],
         "deepth": [2,  5]}
        )

    rf_bo.maximize(n_iter=num_of_iteration)
    # pd.DataFrame(rf_bo.res).to_csv(BO_root+get_related_path(Material_ID))
    pd.DataFrame(data_record).to_csv(BO_routing + get_related_path(Material_ID))
    data_record["trainning_time_consume(s)"].clear()
    data_record["test_time_consume(s)"].clear()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    # for i in range(rank, 127, size):
    #     print(i)
    run_bayes_optimize(10,126)



    # for i in range(119, 127, size):
    #     run_bayes_optimize(100,i)

#     all_data = generate_data.multicsv_data_generater(data_path, return_type="Dataloader")
#
#     for i in range(len(all_data) - rank - 1, -1, -size):
#         Material_ID=all_data.materials[i]
#         print(Material_ID, check_IDexist(Material_ID, result_path))
#
#         if not check_IDexist(Material_ID, result_path):
#             batchsize_list = [64, 128, 512]
#             learning_rate_list = [0.001, 0.002, 0.005]
#             for batchsize in batchsize_list:
#                 for learning_rate in learning_rate_list:
#
#                     all_data.batch_size=batchsize
#                     train_loader, test_loader, Material_ID = all_data[i]
#
#                     my_model = ArtificialNN.simple_ANN(Material_ID)
#                     #         model.load_state_dict(torch.load(model_path))
#
#                     criterion = nn.MSELoss()
#                     #         criterion=loss_function.My_Mass_Balance_loss(1,1)
#                     optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
#
#
#                     ANN_train = DeepNetwork_Train(my_model, criterion, optimizer, train_loader,
#                                                   "." + os.sep + f"{type(criterion).__name__}_data" + os.sep + "mix_" + str(
#                                                       len(
#                                                           Material_ID)) + os.sep + str(Material_ID),test_loader,f"lr={learning_rate}bs={batchsize}")
#
#                     ANN_train.Train(epoch=1)
#
#                     model_paths = "." + os.sep + "saved_model" + os.sep + "simpleANN" + f"lr={learning_rate}bs={batchsize}" + type(
#                         criterion).__name__ + ".pt"
#                     torch.save(my_model.state_dict(), model_paths)
#
