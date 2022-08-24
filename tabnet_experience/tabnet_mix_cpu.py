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

data_set_index = [2]
mix_index = "all"
device = "cpu"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_model = True
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


def get_range(mix_index):
    """
    use to fine target mixture
    :param mix_index: "all" or int range from 1-7
    :return:
    """
    if (mix_index == "all"):
        return 0, 127
    assert mix_index > 0
    start = 0
    end = comb(7, 1)

    for i in range(1, mix_index):
        start += comb(7, i)
        end += comb(7, i + 1)

    return int(start), int(end)


param_grid = [
    # try combinations of hyperparameters
    {'subsample': [0.2, 0.6, 1.0],
     'learning_rate': [0.01, 0.05, 0.1],
     'n_estimators': [300, 400, 500],
     'max_depth': [3, 5, 10],
     'colsample_bytree': [0.6],
     'reg_lambda': [10]}
]


def grid_i(X_train, y_train):
    # train across 3 folds
    grid_search = GridSearchCV(TabNetRegressor(objective='reg:squarederror', n_jobs=3, random_state=42),
                               param_grid,
                               cv=3,
                               scoring='neg_mean_squared_error',
                               return_train_score=True,
                               verbose=1,
                               n_jobs=2)

    start = time.time()
    grid_search.fit(X_train, y_train)
    print("Run time = ", time.time() - start)
    return grid_search


# @Misc{,
#     author = {Fernando Nogueira},
#     title = {{Bayesian Optimization}: Open source constrained global optimization tool for {Python}},
#     year = {2014--},
#     url = " https://github.com/fmfn/BayesianOptimization"
# }
# import os
#
# data_path = ".." + os.sep + "cleaned_data" + os.sep
# result_path = "." + os.sep + "result_data" + os.sep
#
#

from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error

#

data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": [],"epochs":[]}

import argparse
# print(a)
from mpi4py import MPI


def grid_i(X_train, y_train):
    grid_search = GridSearchCV(TabNetRegressor(objective='reg:squarederror', n_jobs=1, random_state=42),
                               param_grid,
                               cv=3,
                               scoring='neg_mean_squared_error',
                               return_train_score=True,
                               verbose=1,
                               n_jobs=2)

    start = time.time()
    grid_search.fit(X_train, y_train)
    print("Run time = ", time.time() - start)
    return grid_search

from sklearn.model_selection import KFold
import numpy as np



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



    train_time = []
    test_time = []
    MSE_loss = []

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    print("totalsize", X.shape)
    kf = KFold(n_splits=4, shuffle=True, random_state=12346)

    for train_index, test_index in kf.split(X):
        print("train_size", train_index.shape, "test_size", test_index.shape)
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
        model_instance.fit(X[train_index], y[train_index],max_epochs=epochs)
        train_time.append(time.time() - start_train)
        start_pred = time.time()
        pred = model_instance.predict(X[test_index])
        test_time.append(time.time() - start_pred)

        MSE_loss.append(mean_squared_error(pred, y[test_index]))

        break

    loss = np.mean(MSE_loss)
    if save_model:
        model_instance.save_model(model_save_path + get_related_path(Material_ID).replace(".csv", ""))

    print("test_time", test_time)
    data_record["trainning_time_consume(s)"].append(np.mean(train_time))
    data_record["test_time_consume(s)"].append(np.mean(test_time))
    data_record["epochs"].append(epochs)

    return -loss


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
    data_record["epochs"].clear()

def run_Grid_search(num_of_iteration):
    print(num_of_iteration)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start, end = get_range(mix_index)

    product_index = list(itertools.product(data_set_index, list(range(start, end))))
    print(product_index)
    print("total size", len(product_index))
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
        run_bayes_optimize(1, 2)
