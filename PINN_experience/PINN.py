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
from model.my_loss_function import My_Mass_Balance_loss_modified
from model import ArtificialNN
import pandas as pd
import numpy as np
data_set_index = [0]
mix_index = "all"
device = "cuda"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_model = False
save_data = False

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


result_path = "." + os.sep + "MSELoss_data" + os.sep

from mpi4py import MPI


# model=ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.simple_ANN,{"material":Material_ID})
# model.fit(Data_loader=train_loader,epoch=10)
# print(model.get_data)
# model.score(test_loader)
# print(model.get_data)

def get_related_path(Material_ID):
    return "mix_" + str(len(Material_ID)) + os.sep + str(Material_ID) + ".csv"


from bayes_opt import BayesianOptimization

class time_clock:
    def __init__(self,name):
        self.name=name
        self.start_times=[]
        self.end_times=[]

    def start_record(self):
        self.start_times.append(time.time())

    def end_record(self):
        self.end_times.append(time.time())

    def save(self,path):
        pd.DataFrame({"start":self.start_times,"end":self.end_times}).to_csv(path)

data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}
import time
def model_cv(**kwargs):
    model_kwargs = {}
    model_kwargs["Nodes_per_layer"] = 300
    model_kwargs["deepth"] = 3
    model_kwargs["material"] = Material_ID[0]
    model_instance = ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.simple_ANN, model_kwargs
                                                             )

    PINN_loss = My_Mass_Balance_loss_modified(kwargs["My_Mass_Balance_loss"],1)



    start_train = time.time()
    model_instance.fit(X_train, y_train, epoch=25, my_criterion=PINN_loss)
    train_time = time.time() - start_train

    score = model_instance.score(X_test, y_test)

    start_pred = time.time()
    model_instance.predict(X_test)
    test_time = time.time() - start_pred


    data_record["trainning_time_consume(s)"].append(train_time)
    data_record["test_time_consume(s)"].append(test_time)

    return -score


import argparse
# print(a)
from mpi4py import MPI


def run_bayes_optimize(num_of_iteration=1, data_index=10):
    dataset="."+os.sep+"mini_data_1"+os.sep
    BO_result_data = "." + os.sep + "BO_result_data" + os.sep
    BO_training_routing = "." + os.sep + "BO_training_routing" + os.sep
    global X_train, y_train, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]
    rf_bo = BayesianOptimization(
        model_cv,
        {'My_Mass_Balance_loss': [0.01, 100]}
    )

    rf_bo.maximize(n_iter=num_of_iteration)
    # pd.DataFrame(rf_bo.res).to_csv(dataset+BO_result_data + get_related_path(Material_ID))
    result_data_root = "." + os.sep + "BO_result_data" + os.sep
    routing_data_root = "." + os.sep + "BO_training_routing" + os.sep
    pd.DataFrame(data_record)
    routing_data_root + get_related_path(Material_ID)
    if save_data:
        if os.path.exists(saved_root + result_data_root + get_related_path(Material_ID)):
            pd.concat([pd.DataFrame(rf_bo.res),
                       pd.read_csv(saved_root + result_data_root + get_related_path(Material_ID), index_col=0,
                                   comment="#")], ignore_index=True).to_csv(
                saved_root + result_data_root + get_related_path(Material_ID))
            f = open(saved_root + result_data_root + get_related_path(Material_ID), "a+")
            f.write(f"# {device}")
            f.close()
            pd.concat([pd.DataFrame(data_record),
                       pd.read_csv(saved_root + routing_data_root + get_related_path(Material_ID), index_col=0,
                                   comment="#")], ignore_index=True).to_csv(
                saved_root + routing_data_root + get_related_path(Material_ID))
            f = open(saved_root + routing_data_root + get_related_path(Material_ID), "a+")
            f.write(f"# {device}")
            f.close()
        else:
            pd.DataFrame(rf_bo.res).to_csv(saved_root + result_data_root + get_related_path(Material_ID))
            f = open(saved_root + result_data_root + get_related_path(Material_ID), "a+")
            f.write(f"# {device}")
            f.close()
            pd.DataFrame(data_record).to_csv(saved_root + routing_data_root + get_related_path(Material_ID))
            f = open(saved_root + routing_data_root + get_related_path(Material_ID), "a+")
            f.write(f"# {device}")
            f.close()
    data_record["trainning_time_consume(s)"].clear()
    data_record["test_time_consume(s)"].clear()



def model_gs(args):
    num, X_train, y_train, X_test, y_test, Material_ID,root = args
    model_kwargs = {}
    model_kwargs["Nodes_per_layer"] = 300
    model_kwargs["deepth"] = 3

    model_kwargs["material"] = Material_ID
    model_instance = ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.simple_ANN, model_kwargs)

    PINN_loss = My_Mass_Balance_loss(num,1)

    #train and record
    start_train = time.time()
    model_instance.fit(X_train, y_train, epoch=30, criterion=PINN_loss)
    train_time = time.time() - start_train


    #test and record
    start_pred = time.time()
    score = model_instance.score(X_test, y_test)
    test_time = time.time() - start_pred

    # data_record["trainning_time_consume(s)"].append(train_time)
    # data_record["test_time_consume(s)"].append(test_time)

    epoch_root = "." + os.sep + "GS_epoch_routing" + os.sep
    pd.DataFrame(model_instance.data_record).to_csv(root+epoch_root + get_related_path(Material_ID))


    return -score ,train_time,test_time


from multiprocessing import Pool


def run_grid_search(data_index=10):
    GS_root = "." + os.sep + "GS_result_data" + os.sep
    GS_routing = "." + os.sep + "GS_training_routing" + os.sep

    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]
    params = np.concatenate([np.linspace(0.05, 0.45, 40), np.linspace(0.45, 100, 20)], axis=0)

    X_train_list=[X_train]*len(params)
    y_train_list=[y_train]*len(params)
    X_test_list=[X_test]*len(params)
    y_test_list=[y_test]*len(params)
    root_list = [saved_root] * len(params)
    material_ID_list = [Material_ID] * len(params)


    args = list(zip(params,X_train_list,y_train_list,X_test_list,y_test_list, material_ID_list,root_list))


    with Pool(6) as p:
        paral_result = np.array(p.map(model_gs, args))

    result,train_time,test_time=paral_result[:,0],paral_result[:,1],paral_result[:,2]



    data_record["trainning_time_consume(s)"]=train_time
    data_record["test_time_consume(s)"]=test_time


    pd.DataFrame({"target": result, "params": params}).to_csv(saved_root+GS_root + get_related_path(Material_ID))
    print(Material_ID)

    pd.DataFrame(data_record).to_csv(saved_root+GS_routing + get_related_path(Material_ID))


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()



    for index in range(rank, len(data_set_index), size):
        data_index = data_set_index[index]
        print(data_index)

        model_save_path = "." + os.sep + "saved_model" + os.sep + "mini_dataset_mixture" + os.sep + f"mini_data_{data_index}" + os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root + f"mini_data_{data_index}" + os.sep
        saved_root = "." + os.sep + "mini_cleaned_data_mixture_"+device + os.sep + f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(100, 2)

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
