import sys

sys.path.append("..")
from data import generate_data
import pandas as pd
from mpi4py import MPI
All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
import time
import os
import itertools
from lightgbm import LGBMRegressor
data_set_index = [0,1,2,3]
mix_index="all"
device = "cpu"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep

save_model=False
save_data=True

data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}

def get_related_path(Material_ID):
    print(Material_ID)
    print(type(Material_ID))
    if isinstance(Material_ID,list):
        return "mix_"+str(len(Material_ID[0]))+os.sep+"mixture.csv"
    else:
        return "mix_"+str(len(Material_ID))+os.sep+str(Material_ID)+".csv"
    print("func:get_related_path  problem")
    raise RuntimeError


from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
#
from sklearn.model_selection import KFold
import numpy as np


def model_cv(**kwargs):

    subsample=kwargs["subsample"] if "subsample" in kwargs.keys() else 0.5
    learning_rate=kwargs["learning_rate"] if "learning_rate" in kwargs.keys() else 0.05
    n_estimators=kwargs["n_estimators"] if "n_estimators" in kwargs.keys() else 400
    max_depth=kwargs["min_samples_split"] if "min_samples_split" in kwargs.keys() else 12
    colsample_bytree = kwargs["colsample_bytree"] if "colsample_bytree" in kwargs.keys() else 0.8
    reg_lambda = kwargs["reg_lambda"] if "reg_lambda" in kwargs.keys() else 15

    lgbm_fit_params = {
        'early_stopping_rounds': 100,
        'eval_set': [(X_val, y_val)],
    }

    print("train_shape",X_train.shape,"val_shape",X_val.shape,"test_shape",X_test.shape)
    train_time = []
    test_time = []
    MSE_loss = []
    patience=100
    if True:
        model_instances = [LGBMRegressor(
            subsample=subsample,
            learning_rate=learning_rate, # float
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
            n_jobs=1)
         for i in range(y_val.shape[-1])]
        start_train = time.time()

        for i, model_instance in enumerate(model_instances):
            model_instance.fit(X_train, y_train[...,i],None,early_stopping_rounds= 100,eval_set= [(X_val, y_val[...,i])],verbose=10)

        train_time.append(time.time() - start_train)
        start_pred = time.time()
        pred = np.array([model_instance.predict(X_test) for model_instance in model_instances]).T
        test_time.append(time.time() - start_pred)

        MSE_loss.append(mean_squared_error(pred, y_test))
    loss = np.mean(MSE_loss)
    if save_model:
        for i, model_instance in enumerate(model_instances):
            model_instance.save_model(model_save_path + get_related_path(Material_ID).replace(".csv", "")+str(i))

    print("test_time", test_time)
    data_record["trainning_time_consume(s)"].append(np.mean(train_time))
    data_record["test_time_consume(s)"].append(np.mean(test_time))

    return -loss

import argparse
# print(a)
from mpi4py import MPI
from sklearn.model_selection import train_test_split
import sklearn
def run_bayes_optimize(num_of_iteration=10,data_index=10):
    BO_root="."+os.sep+"BO_result_data"+os.sep

    global X_train, y_train,X_val, y_val, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=4)
    print("train_shape",X_train.shape,"val_shape",X_val.shape)

    rf_bo = BayesianOptimization(
            model_cv,
        {'subsample': [0.2, 1.0],
         'learning_rate': [0.01,  0.1],
         'n_estimators': [300,  500],
         'max_depth': [3, 10],
         'colsample_bytree': [0.6,0.99],
         'reg_lambda': [10,30]}
        )

    rf_bo.maximize(n_iter=num_of_iteration)
    result_data_root = "." + os.sep + "BO_result_data" + os.sep
    routing_data_root = "." + os.sep + "BO_training_routing" + os.sep
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
        data_index=data_set_index[index]
        print(data_index)

        model_save_path="."+os.sep+"saved_model"+os.sep+"mini_dataset_mixture"+os.sep+f"mini_data_{data_index}"+ os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root+ f"mini_data_{data_index}" + os.sep
        saved_root ="."+os.sep+"mini_cleaned_data_mixture_"+device+"_noPreprocess"+os.sep+ f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(1, 2)

    # all_data = generate_data.multicsv_data_generater(data_path)
    #
    # for i in range(len(all_data)-rank-1,-1,-size):
    #
    #
    #     X_train, y_train, X_test, y_test, Material_ID = all_data[i]
    #     print(Material_ID, check_IDexist(Material_ID, result_path))
    #     print(Material_ID, check_IDexist(Material_ID, result_path))
    #     if not check_IDexist(Material_ID, result_path):
    #         Intermediate_path = "mix_" + str(len(Material_ID)) + os.sep
    #         print(Material_ID)
    #         grid_search_i = grid_i(X_train, y_train)
    #
    #         print(f"best parameters: {grid_search_i.best_params_}")
    #         print(
    #             f"best score:      {-grid_search_i.best_score_:0.5f} (+/-{grid_search_i.cv_results_['std_test_score'][grid_search_i.best_index_]:0.5f})")
    #         results_dfi = pd.DataFrame(grid_search_i.cv_results_)
    #         results_dfi.to_csv(result_path + Intermediate_path + str(Material_ID) + ".csv")

