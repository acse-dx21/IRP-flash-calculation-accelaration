import sys

sys.path.append("..")
from data import generate_data
import pandas as pd
from mpi4py import MPI
import itertools
from sklearn.model_selection import GridSearchCV
import time
import os
from pytorch_tabnet.tab_model import TabNetRegressor
import torch.optim as optim

data_set_index = [0,1,2,3]
mix_index = "all"
device = "cpu"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_model = False
save_data = True
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_related_path(Material_ID):
    print(Material_ID)
    print(type(Material_ID))
    if isinstance(Material_ID, list):
        return "mix_" + str(len(Material_ID[0])) + os.sep + "mixture.csv"
    else:
        return "mix_" + str(len(Material_ID)) + os.sep + str(Material_ID) + ".csv"
    print("func:get_related_path  problem")
    raise RuntimeError



from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error

#

data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": [], "epochs": []}

import argparse
# print(a)
from mpi4py import MPI
import sklearn
from sklearn.model_selection import KFold
import numpy as np


def model_cv(**kwargs):
    n_d = kwargs["n_d"] if "n_d" in kwargs.keys() else 64
    n_a = kwargs["n_a"] if "n_a" in kwargs.keys() else 128
    n_steps = kwargs["n_steps"] if "n_steps" in kwargs.keys() else 1
    gamma = kwargs["gamma"] if "gamma" in kwargs.keys() else 1.3
    print(n_d)
    train_time = []
    test_time = []
    MSE_loss = []

    print("train_shape", X_train.shape, "val_shape", X_val.shape)
    epochs = 1000
    patience = 100
    if True:
        model_instance = TabNetRegressor(
            int(n_d),
            int(n_a),
            int(n_steps),
            int(gamma),
            lambda_sparse=0,
            n_independent=2,
            n_shared=1,
            optimizer_fn=optim.Adam,
            optimizer_params=dict(lr=3e-2, weight_decay=1e-5),
            mask_type="entmax",
            scheduler_params=dict(
                mode="min", patience=5, min_lr=1e-5, factor=0.9),
            scheduler_fn=ReduceLROnPlateau,
            seed=25,
            verbose=10,
            device_name=device
        )
        start_train = time.time()
        model_instance.fit(X_train, y_train, eval_set=[(X_val, y_val)], patience=patience, max_epochs=epochs)
        train_time.append(time.time() - start_train)
        start_pred = time.time()
        pred = model_instance.predict(X_test)
        print(pred,pred.shape)
        test_time.append(time.time() - start_pred)
        MSE_loss.append(mean_squared_error(pred, y_test))
    loss = np.mean(MSE_loss)
    if save_model:
        model_instance.save_model(model_save_path + get_related_path(Material_ID).replace(".csv", ""))

    print("test_time", test_time)
    data_record["trainning_time_consume(s)"].append(np.mean(train_time))
    data_record["test_time_consume(s)"].append(np.mean(test_time))
    data_record["epochs"].append(epochs)

    return -loss


from sklearn.model_selection import train_test_split


def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train, X_val, y_val, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=4)

    print(model_save_path + get_related_path(Material_ID).replace(".csv", ".json"))
    print("run ", num_of_iteration, "times")
    rf_bo = BayesianOptimization(
        model_cv,
        {
            "n_d": [32, 128],
            "n_a": [32, 512],
            "n_steps": [1, 5],
            "gamma": [0.5, 2],
        }
    )

    rf_bo.maximize(init_points=2, n_iter=num_of_iteration)

    # save data
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
    data_record["epochs"].clear()


def run_Grid_search(num_of_iteration):
    print(num_of_iteration)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for index in range(rank, len(data_set_index), size):
        data_index = data_set_index[index]
        print(data_index)

        model_save_path = "." + os.sep + "saved_model" + os.sep + "mini_dataset_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root + f"mini_data_{data_index}" + os.sep
        saved_root = "." + os.sep + "mini_cleaned_data_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        print(mini_data_path, os.listdir(mini_data_path))
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(1, 2)
