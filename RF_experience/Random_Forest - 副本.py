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
data_set_index = [0,1,2,3]
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

from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error

data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}

from mpi4py import MPI


from sklearn.model_selection import KFold
import numpy as np
def model_cv(**kwargs):



    train_time = []
    test_time = []
    MSE_loss = []

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    print("totalsize", X.shape)
    kf = KFold(n_splits=4, shuffle=True, random_state=12346)
    for train_index, test_index in kf.split(X):
        print("train_size", train_index.shape, "test_size", test_index.shape)
        model_instance = RandomForestRegressor(
            n_estimators=int(kwargs['n_estimators']),

            min_samples_split=int(kwargs['min_samples_split']),
            min_impurity_decrease=kwargs['min_impurity_decrease'],
            max_depth=int(kwargs['max_depth'])
        )
        start_train = time.time()
        model_instance.fit(X[train_index], y[train_index])
        train_time.append(time.time() - start_train)
        start_pred = time.time()
        pred = model_instance.predict(X[test_index])
        test_time.append(time.time() - start_pred)

        MSE_loss.append(mean_squared_error(pred, y[test_index]))
    loss = np.mean(MSE_loss)

    print("test_time", test_time)
    data_record["trainning_time_consume(s)"].append(np.mean(train_time))
    data_record["test_time_consume(s)"].append(np.mean(test_time))

    return -loss
from sklearn.model_selection import train_test_split

def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train,X_val, y_val, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=4)

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
        run_bayes_optimize(5, 2)