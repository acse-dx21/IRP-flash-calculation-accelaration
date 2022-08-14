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

import pandas as pd
import numpy as np
import itertools

from itertools import combinations
from scipy.special import comb, perm
from sklearn.model_selection import GridSearchCV
import time
from pytorch_tabnet.tab_model import TabNetRegressor
import torch.optim as optim

data_set_index = [3,]
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
num=50
temp = 500
pressure = 30*100000
row_data = [list(np.linspace(0.5, 1, num) * temp), np.linspace(0.2, 1, num) * pressure]
from torch.optim.lr_scheduler import ReduceLROnPlateau
from bayes_opt import BayesianOptimization
data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}
def model_cv(**kwargs):
    n_d = kwargs["n_d"] if "n_d" in kwargs.keys() else 64
    n_a = kwargs["n_a"] if "n_a" in kwargs.keys() else 128
    n_steps = kwargs["n_steps"] if "n_steps" in kwargs.keys() else 1
    gamma = kwargs["gamma"] if "gamma" in kwargs.keys() else 1.3
    lambda_sparse = kwargs["lambda_sparse"] if "lambda_sparse" in kwargs.keys() else 0
    n_independent = kwargs["n_independent"] if "n_independent" in kwargs.keys() else 2
    n_shared = kwargs["n_shared"] if "n_shared" in kwargs.keys() else 1
    print(n_d)

    epochs=250
    print("here",device)
    model_instance = TabNetRegressor(
        int(n_d),
        int(n_a),
        int(n_steps),
        int(gamma),
        lambda_sparse=0,
        n_independent=2,
        n_shared=1,
        optimizer_fn=optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        mask_type="entmax",
        scheduler_params=dict(
            mode="min", patience=5, min_lr=1e-5, factor=0.9),
        scheduler_fn=ReduceLROnPlateau,
        seed=25,
        verbose=10,
    device_name=device
    )

    start_train = time.time()
    model_instance.fit(X_train, y_train,max_epochs=epochs)
    train_time = time.time() - start_train


    conti_pred=ArtificialNN.continues_pred(model_instance)
    conti_pred.continues_predict(relate_data, row_data[0], row_data[1])

    print("success")
    continues_pred="." + os.sep + "continues_pred" + os.sep
    pd.DataFrame(conti_pred.continues_predict_data).to_csv(saved_root+continues_pred+ get_related_path(Material_ID))


    return conti_pred.continues_predict_data["continues_predict_loss"][-1]

import argparse
# print(a)
from mpi4py import MPI



def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]

    print(model_save_path + get_related_path(Material_ID).replace(".csv", ".json"))

    rf_bo = BayesianOptimization(
        model_cv,
        {
            "n_d": [32, 128],
            "n_a":[32,512],
            "n_steps":[1,5],
            "gamma":[0.5,2],
            "lambda_sparse":[0.1,1],
            "n_independent":[1.5,4.5],
            "n_shared":[0.5,2.5]

        }
    )

    rf_bo.maximize(init_points=1,n_iter=num_of_iteration)

    #save data
    result_data_root = "." + os.sep + "BO_result_data" + os.sep
    routing_data_root = "." + os.sep + "BO_training_routing" + os.sep
    pd.DataFrame(data_record)
    routing_data_root + get_related_path(Material_ID)
    if save_data:
        if os.path.exists(saved_root + result_data_root + get_related_path(Material_ID)):
            pd.concat([pd.DataFrame(rf_bo.res),pd.read_csv(saved_root + result_data_root + get_related_path(Material_ID),index_col=0,comment="#")],ignore_index=True).to_csv(saved_root + result_data_root + get_related_path(Material_ID))
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



if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    start,end = get_range(mix_index)

    product_index = list(itertools.product(data_set_index, list(range(start,end))))
    print(product_index)
    print("total size",len(product_index))
    for index in range(rank, len(data_set_index), size):
        data_index = data_set_index[index]
        print(data_index)

        model_save_path = "." + os.sep + "saved_model" + os.sep + "mini_dataset_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root + f"mini_data_{data_index}" + os.sep
        saved_root = "." + os.sep + "mini_cleaned_data_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(1, 2)


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
