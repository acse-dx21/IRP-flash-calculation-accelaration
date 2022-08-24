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

data_set_index = [0]
mix_index="all"
device = "cuda"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_model=False
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

from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error

#

data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}
from mpi4py import MPI


def compute_material_fraction(out,material_num):
        return (out[..., :material_num].T * out[..., -2] + out[..., material_num:2 * material_num].T * out[..., -1]).T

def model_cv(**kwargs):
    model_save_path="."+os.sep+"saved_model"+os.sep+"mini_dataset_mixture"+os.sep+"mini_data_2"+ os.sep
    model_instance = XGBRegressor()
    model_instance.load_model(model_save_path+get_related_path(Material_ID).replace(".csv",".json"))
    loss_data={"target":[]}
    data_saved_root = saved_root + "continues_pred" + os.sep + "mix_2" + os.sep + "mixture.csv"
    for i in range(1000):
        pred = model_instance.predict(X_test)

        loss = mean_squared_error(pred, y_test)
        loss_data["target"].append(loss)
        X_test[...,-2:]=compute_material_fraction(pred,2)
        print(i,loss)
    exit(0)
    # pd.DataFrame(loss_data).to_csv(data_saved_root)


    return -mean_squared_error(pred, y_test)


def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]
    print(X_test.shape)
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