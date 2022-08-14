import sys

sys.path.append("..")
from data import generate_data
import pandas as pd
from mpi4py import MPI
import itertools
from sklearn.model_selection import GridSearchCV
import time
import os
from xgboost import XGBRegressor

data_set_index = [3,4,5]
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

data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}





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

def model_cv(**kwargs):
    subsample = kwargs["subsample"] if "subsample" in kwargs.keys() else 0.5
    learning_rate = kwargs["learning_rate"] if "learning_rate" in kwargs.keys() else 0.05
    n_estimators = kwargs["n_estimators"] if "n_estimators" in kwargs.keys() else 400
    max_depth = kwargs["min_samples_split"] if "min_samples_split" in kwargs.keys() else 12
    colsample_bytree = kwargs["colsample_bytree"] if "colsample_bytree" in kwargs.keys() else 0.8
    reg_lambda = kwargs["reg_lambda"] if "reg_lambda" in kwargs.keys() else 15
    start_train = time.time()
    model_instance = XGBRegressor(
        subsample=subsample,
        learning_rate=learning_rate,  # float
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        n_jobs=1

    ).fit(X_train, y_train)
    train_time = time.time() - start_train
    if save_model:
        model_instance.save_model(model_save_path+get_related_path(Material_ID).replace(".csv",".json"))
    start_pred = time.time()
    pred = model_instance.predict(X_test)
    test_time = time.time() - start_pred
    print("test time",test_time)
    data_record["trainning_time_consume(s)"].append(train_time)
    data_record["test_time_consume(s)"].append(test_time)

    return -mean_squared_error(pred, y_test)


def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]
    print(X_test.shape)
    exit(0)
    print(model_save_path+get_related_path(Material_ID).replace(".csv",".json"))

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
    if save_data:
        pd.DataFrame(data_record).to_csv(saved_root + routing_data_root + get_related_path(Material_ID))
        pd.DataFrame(rf_bo.res).to_csv(saved_root + result_data_root + get_related_path(Material_ID))

    data_record["trainning_time_consume(s)"].clear()
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
        saved_root = "."+os.sep+"mini_cleaned_data_mixture_"+device+os.sep+ f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(10, 2)