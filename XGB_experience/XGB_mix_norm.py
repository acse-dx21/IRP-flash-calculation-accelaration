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

data_set_index = [0,1,2,3]

mix_index = "all"
device = "cuda"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_model = False
save_data = True


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


data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}
from sklearn.preprocessing import normalize


def normalization(ypred,drop_rata=None):
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

    # data2.loc[data[0] < drop_rata, [0,1,4,5]] =[0,0,0,1]
    # data2.loc[data[2] < drop_rata, [2,3,4,5]] =[0,0,1,0]

    data2.loc[(data2[4] == 0) | (data2[5] == 1), [0, 1]] = 0
    data2.loc[(data2[5] == 0) | (data2[4] == 1), [2, 3]] = 0
    data2 = data2.to_numpy()
    #
    # return np.concatenate([normalize(data2[:, :divide].view(), norm='l1'),
    #                 normalize(data2[:, divide:2 * divide].view(), norm='l1'),
    #                 normalize(data2[:, 2 * divide:].view(), norm='l1')], axis=1)

    return data2


y = np.random.random(size=(10, 6)) / 3

y = normalization(y,0.5)

import argparse
# print(a)
from mpi4py import MPI

from sklearn.model_selection import KFold
import numpy as np


def model_cv(**kwargs):
    subsample = kwargs["subsample"] if "subsample" in kwargs.keys() else 0.5
    learning_rate = kwargs["learning_rate"] if "learning_rate" in kwargs.keys() else 0.05
    n_estimators = kwargs["n_estimators"] if "n_estimators" in kwargs.keys() else 400
    max_depth = kwargs["min_samples_split"] if "min_samples_split" in kwargs.keys() else 12
    colsample_bytree = kwargs["colsample_bytree"] if "colsample_bytree" in kwargs.keys() else 0.8
    reg_lambda = kwargs["reg_lambda"] if "reg_lambda" in kwargs.keys() else 15

    train_time = []
    test_time = []
    MSE_loss = []
    patience=100

    print("train_shape",X_train.shape,"val_shape",X_val.shape,"test_shape",X_test.shape)
    if True:

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
        )
        start_train = time.time()
        model_instance.fit(X_train, y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=patience)
        train_time.append(time.time() - start_train)
        start_pred = time.time()
        pred = model_instance.predict(X_test)
        test_time.append(time.time() - start_pred)
        print("before_reg",mean_squared_error(pred, y_test),"after_reg",mean_squared_error(normalization(pred), y_test))
        MSE_loss.append(mean_squared_error(normalization(pred), y_test))
    loss=np.mean(MSE_loss)
    if save_model:
        model_instance.save_model(model_save_path + get_related_path(Material_ID).replace(".csv", ""))

    print("test_time", test_time)
    data_record["trainning_time_consume(s)"].append(np.mean(train_time))
    data_record["test_time_consume(s)"].append(np.mean(test_time))

    return -loss


from sklearn.model_selection import train_test_split
import sklearn
def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train,X_val, y_val, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]



    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=4)

    print(X_train.shape)
    print(y_train.shape)
    print(model_save_path + get_related_path(Material_ID).replace(".csv", ".json"))

    rf_bo = BayesianOptimization(
        model_cv,
        {'subsample': [0.2, 1.0],
         'learning_rate': [0.01, 0.1],
         'n_estimators': [300, 500],
         'max_depth': [3, 10],
         'colsample_bytree': [0.6, 0.99],
         'reg_lambda': [10, 30]}
    )


    rf_bo.maximize(n_iter=num_of_iteration)
    result_data_root = "." + os.sep + "BO_result_data" + os.sep
    routing_data_root = "." + os.sep + "BO_training_routing" + os.sep
    pd.DataFrame(data_record)
    routing_data_root + get_related_path(Material_ID)

    #save data
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


def run_Grid_search(num_of_iteration):
    print(num_of_iteration)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for index in range(rank, len(data_set_index), size):
        data_index = data_set_index[index]
        print(data_index)

        model_save_path = "." + os.sep + "saved_model" + os.sep + "mini_dataset_mixture" + os.sep + f"mini_data_{data_index}" + os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root + f"mini_data_{data_index}" + os.sep #the training\testing data come from
        saved_root = "." + os.sep + "mini_cleaned_data_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # relate_data.set_return_type("Dataloader")
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(10, 2)
