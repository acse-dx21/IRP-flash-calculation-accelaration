import sys

sys.path.append("..")
import os
from data import generate_data
import tool
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from model import ArtificialNN
from model.train import DeepNetwork_Train
import torch
import torch.nn as nn
import thermo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


import time

data_set_index = [3]
mix_index="all"
device = "cuda"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_data=False
save_model=True


def get_related_path(Material_ID):
    if isinstance(Material_ID,list):
        return "mix_"+str(len(Material_ID[0]))+os.sep+"mixture.csv"
    return "mix_"+str(len(Material_ID))+os.sep+str(Material_ID)+".csv"


from sklearn.model_selection import KFold
import numpy as np
from bayes_opt import BayesianOptimization
data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}
def model_cv(**kwargs):
    kwargs["Nodes_per_layer"] = 500
    kwargs["deepth"] = 5
    kwargs["material"]=Material_ID[0]
    model_instance = ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.simple_ANN,kwargs)
    model_instance.set_device(device)
    train_time = []
    test_time = []
    MSE_loss = []

    print()
    print("train_shape",X_train.shape,"val_shape",X_val.shape)
    if True:

        model_instance = ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.simple_ANN, kwargs)
        model_instance.set_device(device)
        start_train = time.time()
        # model_instance.fit(X_train, y_train,max_epochs=1,patience=100,eval_set=[(X_val,y_val)])
        model_instance.load_model(
            "E:\\Ic document\\IRP_flash_calculation_accelaration\\Simple_ANN_experience\\simpleANN_500.pth")
        train_time.append(time.time() - start_train)
        start_pred = time.time()
        pred = model_instance.predict(X_test)
        print(pred)
        test_time.append(time.time() - start_pred)

        MSE_loss.append(mean_squared_error(pred, y_test))
        print("loss",MSE_loss[-1])
    loss = np.mean(MSE_loss)

    for i in range(100):
        pred = model_instance.predict(np.array([X_test[i,:]]))

    # if save_model:
    #     model_instance.save_model("simpleANN_500.pth")

    exit()
    print("test_time", test_time)
    data_record["trainning_time_consume(s)"].append(np.mean(train_time))
    data_record["test_time_consume(s)"].append(np.mean(test_time))

    return -loss

import argparse
# print(a)
from mpi4py import MPI

import sklearn

def run_bayes_optimize(num_of_iteration=1,data_index=2):

    BO_root="."+os.sep+"BO_result_data"+os.sep
    BO_routing =  "." + os.sep + "BO_training_routing" + os.sep



    global X_train, y_train, X_val, y_val, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]
    preprocess = sklearn.preprocessing.StandardScaler().fit(X_train)


    print("mean",preprocess.mean_)
    print("var",preprocess.var_)
    print("after",preprocess.transform(np.array([[4.8720e+06, 3.3700e+06, 3.0532e+02, 4.6970e+02, 9.8000e-02, 2.5100e-01,
  3.1000e+02 ,1.0000e+06 ,5.0000e-01, 5.0000e-01]])))
    # print(preprocess.get_params())
    print(X_test)
    #normalize it
    X_train = preprocess.transform(X_train)

    X_test = preprocess.transform(X_test)
    print(X_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=4)

    rf_bo = BayesianOptimization(
            model_cv,
        {'Nodes_per_layer': (100, 1000),
         "deepth": (2,  6)}
        )

    rf_bo.maximize(init_points=5,n_iter=num_of_iteration)

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


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for index in range(rank, len(data_set_index), size):
        data_index=data_set_index[index]
        print(data_index)
        model_save_path = "." + os.sep + "saved_model" + os.sep + "mini_dataset_mixture" + os.sep + f"mini_data_{data_index}" + os.sep
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


        run_bayes_optimize(5, 2)
