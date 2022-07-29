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
import itertools

from itertools import combinations
from scipy.special import comb, perm
from sklearn.model_selection import GridSearchCV
import time

data_set_index = [0, 3, 4, 5]
mix_index="all"
device = "cuda"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_data=False
def get_range(mix_index):
    """
    use to fine target mixture
    :param mix_index: "all" or int range from 1-7
    :return:
    """
    if(mix_index=="all"):
        return 0,127
    assert mix_index>0
    start=0
    end=comb(7,1)

    for i in range(1,mix_index):
        start+=comb(7,i)
        end+=comb(7,i+1)

    return int(start),int(end)




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
    if isinstance(Material_ID,list):
        return "mix_"+str(len(Material_ID[0]))+os.sep+"mixture.csv"
    return "mix_"+str(len(Material_ID))+os.sep+str(Material_ID)+".csv"



from bayes_opt import BayesianOptimization
data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}
def model_cv(**kwargs):
    kwargs["Nodes_per_layer"] = int(kwargs["Nodes_per_layer"])
    kwargs["deepth"] = int(kwargs["deepth"])
    kwargs["material"]=Material_ID[0]
    model_instance = ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.simple_ANN,kwargs)
    model_instance.set_device(device)
    print(model_instance)
    # record_data
    start_train = time.time()
    model_instance.fit(X_train,y_train,epoch=10)
    train_time = time.time() - start_train

    score=model_instance.score(X_test,y_test)

    start_pred = time.time()
    pred = model_instance.predict(X_test)
    test_time = time.time() - start_pred

    data_record["trainning_time_consume(s)"].append(train_time)
    data_record["test_time_consume(s)"].append(test_time)

    epoch_root="."+os.sep+"BO_epoch_routing"+os.sep
    pd.DataFrame(model_instance.data_record).to_csv(saved_root+epoch_root + get_related_path(Material_ID))

    f = open(saved_root+epoch_root + get_related_path(Material_ID), "a+")
    f.write(f"# {device}")
    f.close()

    return -score

import argparse
# print(a)
from mpi4py import MPI



def run_bayes_optimize(num_of_iteration=1,mix=2):

    BO_root="."+os.sep+"BO_result_data"+os.sep
    BO_routing =  "." + os.sep + "BO_training_routing" + os.sep

    global X_train, y_train, X_test, y_test, Material_ID

    X_train, y_train, X_test, y_test, Material_ID = relate_data[mix]
    rf_bo = BayesianOptimization(
            model_cv,
        {'Nodes_per_layer': (100, 1000),
         "deepth": (2,  6)}
        )

    rf_bo.maximize(init_points=5,n_iter=num_of_iteration)
    if save_data:
        pd.DataFrame(rf_bo.res).to_csv(saved_root+BO_root+get_related_path(Material_ID))
        f=open(saved_root+BO_root+get_related_path(Material_ID),"a+")
        f.write(f"# {device}")
        f.close()
        pd.DataFrame(data_record).to_csv(saved_root+BO_routing+ get_related_path(Material_ID))
        f = open(saved_root+BO_routing+ get_related_path(Material_ID), "a+")
        f.write(f"# {device}")
        f.close()
    data_record["trainning_time_consume(s)"].clear()
    data_record["test_time_consume(s)"].clear()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    start,end = get_range(mix_index)

    product_index = list(itertools.product(data_set_index, list(range(start,end))))
    print(product_index)
    print("total size",len(product_index))
    for index in range(rank, len(data_set_index), size):
        data_index=data_set_index[index]
        print(data_index)

        mini_data_path = ".." + os.sep + "data" + os.sep + data_root+ f"mini_data_{data_index}" + os.sep
        saved_root = "." + os.sep + "mini_cleaned_data_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")

        # print("compute mixture ", str(mix_index))
        #
        # run_bayes_optimize(45, i)


        run_bayes_optimize(10, 2)


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
