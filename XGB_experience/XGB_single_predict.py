import sys

sys.path.append("..")
from data import generate_data
import pandas as pd
import itertools
from sklearn.model_selection import GridSearchCV
import time
import os
from xgboost import XGBRegressor

data_set_index = [0]
mix_index="all"
device = "cuda"
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

data_record = {"test_time_consume(s)": []}

# print(a)
from mpi4py import MPI

def model_cv(**kwargs):
    subsample = kwargs["subsample"] if "subsample" in kwargs.keys() else 0.5
    learning_rate = kwargs["learning_rate"] if "learning_rate" in kwargs.keys() else 0.05
    n_estimators = kwargs["n_estimators"] if "n_estimators" in kwargs.keys() else 400
    max_depth = kwargs["min_samples_split"] if "min_samples_split" in kwargs.keys() else 12
    colsample_bytree = kwargs["colsample_bytree"] if "colsample_bytree" in kwargs.keys() else 0.8
    reg_lambda = kwargs["reg_lambda"] if "reg_lambda" in kwargs.keys() else 15
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
    if save_model:
        model_instance.save_model(model_save_path+get_related_path(Material_ID).replace(".csv",".json"))
    for i in range(100):
        start_pred = time.time()
        pred = model_instance.predict([X_test[i, :]])
        test_time = time.time() - start_pred

        data_record["test_time_consume(s)"].append(test_time)
    print("test time",test_time)



    data_record["test_time_consume(s)"].append(test_time)

    return -1


def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]
    print("train_size",X_train.shape,"test_size",X_test.shape)
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

    rf_bo.maximize(n_iter=num_of_iteration,init_points=1)
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


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for index in range(rank, len(data_set_index), size):
        data_index=data_set_index[index]
        print(data_index)

        model_save_path="."+os.sep+"saved_model"+os.sep+"mini_dataset_mixture"+os.sep+f"mini_data_{data_index}"+ os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root+ f"mini_data_{data_index}" + os.sep
        saved_root = "."+os.sep+"mini_cleaned_data_mixture_"+device+os.sep+ "single_predict" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(1, 2)