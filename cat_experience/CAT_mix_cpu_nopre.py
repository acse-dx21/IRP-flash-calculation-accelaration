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
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
data_set_index = [0,1,2,3,]
mix_index="all"
device = "CPU"
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





from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error

#

data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}





import argparse
# print(a)
from mpi4py import MPI



import numpy as np
def model_cv(**kwargs):

    train_time = []
    test_time = []
    MSE_loss = []
    patience=100

    print("train_shape", X_train.shape, "val_shape", X_val.shape, "test_shape", X_test.shape)
    if True:
        model_instances = [CatBoostRegressor(iterations=1000,l2_leaf_reg=kwargs['l2_leaf_reg'],
                                                                depth=int(kwargs['depth']),
                                                                learning_rate=kwargs['learning_rate'],
                                           bagging_temperature=kwargs['bagging_temperature'],
                                                            eval_metric='RMSE',
                                                                loss_function='RMSE',task_type="CPU",leaf_estimation_iterations=int(kwargs['leaf_estimation_iterations'])) for i in range(y_val.shape[-1])]

        start_train = time.time()
        for i, model_instance in enumerate(model_instances):
            model_instance.fit(X_train, y_train[..., i], early_stopping_rounds=patience,
                               eval_set=[(X_val, y_val[..., i])], verbose=10)


        train_time.append(time.time() - start_train)
        start_pred = time.time()
        pred = np.array([model_instance.predict(X_test,task_type=device) for model_instance in model_instances]).T
        print("pred",pred.shape)
        test_time.append(time.time() - start_pred)

        MSE_loss.append(mean_squared_error(pred, y_test))
    loss = np.mean(MSE_loss)
    if save_model:
        for i, model_instance in enumerate(model_instances):
            model_instance.save_model(model_save_path + get_related_path(Material_ID).replace(".csv", "") + str(i))

    print("test_time", test_time)
    data_record["trainning_time_consume(s)"].append(np.mean(train_time))
    data_record["test_time_consume(s)"].append(np.mean(test_time))

    return -loss



from sklearn.model_selection import train_test_split

def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train,X_val,y_val, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=4)
    print("train_size",X_train.shape,"test_size",X_test.shape)
    print(model_save_path+get_related_path(Material_ID).replace(".csv",".json"))

    rf_bo = BayesianOptimization(
        model_cv,
        {
            'l2_leaf_reg':[1,10],
        "learning_rate": [0.05,0.5],
        "depth":[2,16],
         "bagging_temperature" : [0.1,0.5],
        'leaf_estimation_iterations':[1,20]}
    )

    rf_bo.maximize(init_points=5,n_iter=num_of_iteration)
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


    start,end = get_range(mix_index)

    product_index = list(itertools.product(data_set_index, list(range(start,end))))
    print(product_index)
    print("total size",len(product_index))
    for index in range(rank, len(data_set_index), size):
        data_index=data_set_index[index]
        print(data_index)

        model_save_path="."+os.sep+"saved_model"+os.sep+"mini_dataset_mixture"+os.sep+f"mini_data_{data_index}"+ os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root+ f"mini_data_{data_index}" + os.sep
        saved_root = "."+os.sep+"mini_cleaned_data_mixture_cpu_noPreprocess"+os.sep+ f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(1, 2)