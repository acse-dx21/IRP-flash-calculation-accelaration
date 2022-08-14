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

from sklearn.ensemble import RandomForestRegressor
data_set_index = [0]
mix_index="all"
device = "cpu"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_model=True
save_data=True
def get_related_path(Material_ID):
    print(Material_ID)
    print(type(Material_ID))
    if isinstance(Material_ID,list):
        return "mix_"+str(len(Material_ID[0]))+os.sep+"mixture.csv"
    else:
        return "mix_"+str(len(Material_ID))+os.sep+str(Material_ID)+".csv"
    print("func:get_related_path  problem")
    raise RuntimeError


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
    grid_search = GridSearchCV(RandomForestRegressor(objective='reg:squarederror', n_jobs=3, random_state=42),
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

data_record = { "test_time_consume(s)": []}





import argparse
# print(a)
from mpi4py import MPI



def model_cv(**kwargs):

    start_train = time.time()
    model_instance = RandomForestRegressor(
        n_estimators=int(kwargs['n_estimators']),

        min_samples_split= int(kwargs['min_samples_split']),
    min_impurity_decrease=kwargs['min_impurity_decrease'],
    max_depth=int(kwargs['max_depth'])
    ).fit(X_train, y_train)
    train_time = time.time() - start_train

    for i in range(100):
        start_pred = time.time()
        pred = model_instance.predict([X_test[i, :]])
        test_time = time.time() - start_pred

        data_record["test_time_consume(s)"].append(test_time)
    # data_record["epochs"].append(epochs)

    return -1


def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]
    print("train_size",X_train.shape,"test_size",X_test.shape)
    print(model_save_path+get_related_path(Material_ID).replace(".csv",".json"))

    rf_bo = BayesianOptimization(
        model_cv,
        {"n_estimators":[5,300],
         'min_samples_split': [10, 100],
         'min_impurity_decrease': [0, 0.1],
         'max_depth': [3, 50]}

    )

    rf_bo.maximize(n_iter=num_of_iteration)
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
    data_record["test_time_consume(s)"].clear()

def run_Grid_search(num_of_iteration):
    print(num_of_iteration)


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

        model_save_path="."+os.sep+"saved_model"+os.sep+"mini_dataset_mixture"+os.sep+f"mini_data_{data_index}"+ os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root+ f"mini_data_{data_index}" + os.sep
        saved_root = "." + os.sep + "mini_cleaned_data_mixture_" + device + os.sep + f"single_predict" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(3, 2)