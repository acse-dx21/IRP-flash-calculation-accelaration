import sys

sys.path.append("..")
from data import generate_data
import pandas as pd
from mpi4py import MPI
import itertools
from sklearn.model_selection import GridSearchCV
import time
import os
from model import ArtificialNN
import torch.optim as optim
import numpy as np

data_set_index = [0, 1, 2, 3, 4, 5]
mix_index = "all"
device = "cuda"
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

from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error

#

data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": [],"epochs":[]}

import argparse
# print(a)
from mpi4py import MPI

import tensorflow as tf


def prepare_tf_dataset(
        X,
        batch_size,
        y=None,
        shuffle=False,
        drop_remainder=False,
):
    size_of_dataset = len(X)
    if y is not None:
        y = tf.one_hot(y.astype(int), 2)
        ds = tf.data.Dataset.from_tensor_slices((np.array(X.astype(np.float32)), y))
    else:
        ds = tf.data.Dataset.from_tensor_slices(np.array(X.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=size_of_dataset)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    autotune = tf.data.experimental.AUTOTUNE
    ds = ds.prefetch(autotune)
    return ds

num=20
temp = 500
pressure = 30*100000
row_data = [list(np.linspace(0.5, 1, num) * temp), np.linspace(0.2, 1, num) * pressure]

def model_cv(**kwargs):
    kwargs["inner_feature_dim"] = int(kwargs["inner_feature_dim"])
    kwargs["out_feature_dim"] = int(kwargs["out_feature_dim"])
    kwargs["n_step"] = int(kwargs["n_step"])
    kwargs["n_total"] = int(kwargs["n_total"])
    kwargs["n_shared"] = int(kwargs["n_shared"])

    kwargs["input_dim"] = X_train.shape[-1]
    kwargs["output_dim"] = y_train.shape[-1]
    epochs=10

    model_instance = ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.my_tabnet, kwargs)
    model_instance.set_device(device)
    start_train = time.time()

    model_instance.fit(X_train, y_train, epoch=epochs)

    train_time = time.time() - start_train

    print("first_pred",model_instance.score(X_test, y_test))
    conti_pred = model_instance.continues_predict(relate_data, row_data[0], row_data[1])
    print("success")
    exit(0)
    if save_model:
        model_instance.save_model(model_save_path + get_related_path(Material_ID).replace(".csv", ".json"))
    start_pred = time.time()
    pred = model_instance.predict(X_test)
    test_time = time.time() - start_pred

    data_record["trainning_time_consume(s)"].append(train_time)
    data_record["test_time_consume(s)"].append(test_time)
    data_record["epochs"].append(epochs)
    return -mean_squared_error(pred, y_test)


def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]

    print(model_save_path + get_related_path(Material_ID).replace(".csv", ".json"))

    rf_bo = BayesianOptimization(
        model_cv,
        {
            "inner_feature_dim": [50, 400],
            "out_feature_dim": [100, 200],
            "n_total": [4, 8],
            "n_shared": [2, 3],
            "n_step": [2, 5]
        }
    )

    rf_bo.maximize(n_iter=num_of_iteration)
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

    start, end = get_range(mix_index)

    product_index = list(itertools.product(data_set_index, list(range(start, end))))

    for index in range(rank, len(data_set_index), size):
        data_index = data_set_index[index]
        print(data_index)

        model_save_path = "." + os.sep + "saved_model" + os.sep + "mini_dataset_mixture" + os.sep + f"mini_data_{data_index}" + os.sep
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
