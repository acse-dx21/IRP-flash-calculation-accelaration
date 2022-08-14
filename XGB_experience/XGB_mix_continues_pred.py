import sys
import numpy as np

sys.path.append("..")
from data import generate_data
import pandas as pd
from mpi4py import MPI
import itertools
from sklearn.model_selection import GridSearchCV
import time
import os
from xgboost import XGBRegressor
import xgboost as xgb
from model import ArtificialNN
data_set_index = [3]

mix_index = "all"
device = "cuda"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_model = False
save_data = False


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
    grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror', n_jobs=3, random_state=42),
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


from sklearn.preprocessing import normalize


def normalization(ypred,drop_rata):
    result = []
    divide = 2
    data = pd.DataFrame(ypred)
    data2 = pd.DataFrame(ypred)

    data[data.columns[0]] = data[data.columns[0]] + data[data.columns[1]]
    data[data.columns[1]] = data[data.columns[0]]

    data[data.columns[2]] = data[data.columns[2]] + data[data.columns[3]]
    data[data.columns[3]] = data[data.columns[2]]

    data[data.columns[4]] = data[data.columns[4]] + data[data.columns[5]]
    data[data.columns[5]] = data[data.columns[4]]
    # data2=data2.where(data>0.55,0)
    data2.loc[data2[4] > 1, [4, 5]] = [1, 0]
    data2.loc[data2[4] < 0, [4, 5]] = [0, 1]
    data2.loc[data2[5] > 1, [4, 5]] = [0, 1]
    data2.loc[data2[5] < 0, [4, 5]] = [1, 0]

    data2 = data2.where(data2 < 1, 1)
    data2 = data2.where(data2 > 0, 0)

    data2.loc[data[0] < drop_rata, [0,1,4,5]] =[0,0,0,1]
    data2.loc[data[2] < drop_rata, [2,3,4,5]] =[0,0,1,0]

    data2.loc[(data2[4] == 0) | (data2[5] == 1), [0, 1]] = 0
    data2.loc[(data2[5] == 0) | (data2[4] == 1), [2, 3]] = 0
    data2 = data2.to_numpy()
    #
    return np.concatenate([normalize(data2[:, :divide].view(), norm='l1'),
                    normalize(data2[:, divide:2 * divide].view(), norm='l1'),
                    normalize(data2[:, 2 * divide:].view(), norm='l1')], axis=1)

    # return data2


y = np.random.random(size=(10, 6)) / 3

y = normalization(y,0.5)

import argparse
# print(a)
from mpi4py import MPI


def grid_i(X_train, y_train):
    grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror', n_jobs=1, random_state=42),
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


from sklearn.model_selection import train_test_split
num=50
temp = 500
pressure = 30*100000
row_data = [list(np.linspace(0.5, 1, num) * temp), np.linspace(0.2, 1, num) * pressure]
from torch.optim.lr_scheduler import ReduceLROnPlateau
from bayes_opt import BayesianOptimization
data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}

def model_cv(**kwargs):
    subsample = kwargs["subsample"] if "subsample" in kwargs.keys() else 0.922
    learning_rate = kwargs["learning_rate"] if "learning_rate" in kwargs.keys() else 0.074
    n_estimators = kwargs["n_estimators"] if "n_estimators" in kwargs.keys() else 500
    max_depth = kwargs["max_depth"] if "max_depth" in kwargs.keys() else 3
    colsample_bytree = kwargs["colsample_bytree"] if "colsample_bytree" in kwargs.keys() else 0.8
    reg_lambda = kwargs["reg_lambda"] if "reg_lambda" in kwargs.keys() else 15
    drop_rata=kwargs["drop_rata"] if "drop_rata" in kwargs.keys() else 0.5
    start_train = time.time()
    model_instance = XGBRegressor(
        tree_method="gpu_hist",
        subsample=subsample,
        learning_rate=learning_rate,  # float
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        predictor="gpu_predictor",
        n_jobs=1

    ).fit(X_train, y_train)

    train_time = time.time() - start_train


    conti_pred=ArtificialNN.continues_pred(model_instance)
    conti_pred.continues_predict(relate_data, row_data[0], row_data[1])

    print("success")
    continues_pred="." + os.sep + "continues_pred" + os.sep
    pd.DataFrame(conti_pred.continues_predict_data).to_csv(saved_root+continues_pred+ get_related_path(Material_ID))


    return conti_pred.continues_predict_data["continues_predict_loss"][-1]


def run_bayes_optimize(num_of_iteration=1, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]
    print(X_train.shape)
    print(y_train.shape)
    print(model_save_path + get_related_path(Material_ID).replace(".csv", ".json"))

    rf_bo = BayesianOptimization(
        model_cv,
        {'subsample': [0.2, 1.0],
         'max_depth': [3, 10],
         'reg_lambda': [5, 30],
        "drop_rata":[0.3,0.7]}
    )

    rf_bo.maximize(n_iter=num_of_iteration)
    result_data_root = "." + os.sep + "BO_result_data" + os.sep
    routing_data_root = "." + os.sep + "BO_training_routing" + os.sep
    pd.DataFrame(data_record)
    routing_data_root + get_related_path(Material_ID)
    if save_data:
        pd.DataFrame(data_record).to_csv(saved_root + routing_data_root + get_related_path(Material_ID), mode="a+")
        pd.DataFrame(rf_bo.res).to_csv(saved_root + result_data_root + get_related_path(Material_ID), mode="a+")

    data_record["trainning_time_consume(s)"].clear()
    data_record["test_time_consume(s)"].clear()


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
        saved_root = "." + os.sep + "mini_cleaned_data_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # relate_data.set_return_type("Dataloader")
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(1, 2)
