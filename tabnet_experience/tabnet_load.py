import sys

sys.path.append("..")
from data import generate_data
import pandas as pd
from mpi4py import MPI

from sklearn.model_selection import GridSearchCV
import time
import os
from xgboost import XGBRegressor
data_size=3000
data_num = 2
mini_data_path = ".." + os.sep + "data" + os.sep + f"mini_cleaned_data_{data_num}" + os.sep
saved_root = "." + os.sep + f"mini_data_{data_num}" + os.sep
All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
# saved_root="."+os.sep+"complete_dataset"+os.sep  #for complete dataset
relate_data = generate_data.multicsv_data_generater()
# relate_data = generate_data.multicsv_data_generater()
model_save_path = "." + os.sep + "saved_model" + os.sep+f"mini_dataset_{data_num}"+os.sep
save = True
X_train = 0
y_train = 0
X_test = 0
y_test = 0
material_ID = 0

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

time_record = {"load_time_consume(s)": [], "test_time_consume(s)": [],"data_size":[]}
result_record = {"loss_dataset_1": [], "loss_dataset_2": [],"data_size":[]}

def model_cv(X_train, y_train, X_test, y_test,model_path):
    print(data_size, X_train.shape)
    start_train = time.time()
    model_instance = XGBRegressor()
    model_instance.load_model(model_path)
    train_time = time.time() - start_train

    start_pred = time.time()
    pred = model_instance.predict(X_train)
    test_time = time.time() - start_pred

    time_record["load_time_consume(s)"].append(train_time)
    time_record["test_time_consume(s)"].append(test_time)
    time_record["data_size"].append(data_size)

    result_record["loss_dataset_1"].append(-mean_squared_error(pred, y_train))
    result_record["loss_dataset_2"].append(-mean_squared_error(model_instance.predict(X_test), y_test))
    result_record["data_size"].append(data_size)




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


def get_related_path(Material_ID):
    return "mix_" + str(len(Material_ID)) + os.sep + str(Material_ID) + ".csv"

def run_bayes_optimize(num_of_iteration=10, data_index=10):

    global X_train, y_train, X_test, y_test, Material_ID,data_size
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]

    global model_save_path

    model_path=model_save_path+get_related_path(Material_ID).replace(".csv",".json")
    for data_size in range(3000,40000,1000):
        print(data_size,X_train.shape)
        print(time_record)
        model_cv(X_train[:data_size], y_train[:data_size], X_test[:int(data_size/2)], y_test[:int(data_size/2)],model_path)

    result_data_root = "." + os.sep + "result_data" + os.sep
    routing_data_root = "." + os.sep + "training_routing" + os.sep

    if (save):
        pd.DataFrame(time_record).to_csv(saved_root + routing_data_root + get_related_path(Material_ID))
        pd.DataFrame(result_record).to_csv(saved_root + result_data_root + get_related_path(Material_ID))
    time_record["load_time_consume(s)"].clear()
    time_record["test_time_consume(s)"].clear()
    time_record["data_size"].clear()
    result_record["loss_dataset_1"].clear()
    result_record["loss_dataset_2"].clear()
    result_record["data_size"].clear()


def run_Grid_search(num_of_iteration):
    print(num_of_iteration)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    stratigy = "BO"

    parser = argparse.ArgumentParser(description='bayes optimazation(--BO) or grid_search(--GS) ')
    parser.add_argument('--BO', type=int, help='Bayes optimize with number of iteration')
    parser.add_argument('--GS', type=bool, help='type use')

    args = parser.parse_args()

    if args.BO is not None or stratigy == "BO":

        for i in range(rank+92 , 127, size):
            run_bayes_optimize(55, i)
    # elif args.GS is not None or stratigy == "GS":
    #     run_Grid_search(args.GS)

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
