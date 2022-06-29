import sys

sys.path.append("..")
from data import generate_data
import pandas as pd
from mpi4py import MPI
All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']

from sklearn.model_selection import GridSearchCV
import time
import os
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
relate_data=generate_data.multicsv_data_generater()


X_train=0
y_train=0
X_test=0
y_test=0
material_ID=0


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





# import os
#
# data_path = ".." + os.sep + "cleaned_data" + os.sep
# result_path = "." + os.sep + "result_data" + os.sep
#
#

from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
#



def model_cv(**kwargs):
    kwargs["n_estimators"]=int(kwargs["n_estimators"])
    kwargs['max_depth']=int(kwargs['max_depth'])

    model_instance = RandomForestRegressor(
            **kwargs
        ).fit(X_train, y_train)
    return -mean_squared_error(model_instance.predict(X_test), y_test)


import argparse
# print(a)
from mpi4py import MPI




def grid_i(X_train, y_train):
    # train across 3 folds
    grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror', n_jobs=1, random_state=42),
                               param_grid,
                               cv=3,
                               scoring='neg_mean_squared_error',
                               return_train_score=True,
                               verbose=1,
                               n_jobs=1)

    start = time.time()
    grid_search.fit(X_train, y_train)
    print("Run time = ", time.time() - start)
    return grid_search


def get_related_path(Material_ID):
    return "mix_"+str(len(Material_ID))+os.sep+str(Material_ID)+".csv"

def run_bayes_optimize(num_of_iteration=10,data_index=10):
    BO_root="."+os.sep+"BO_result_data"+os.sep
    global X_train, y_train, X_test, y_test, material_ID
    X_train, y_train, X_test, y_test, material_ID = relate_data[data_index]
    rf_bo = BayesianOptimization(
            model_cv,
        {
         'n_estimators': [300,  500],
         'max_depth': [3, 10]}
        )

    rf_bo.maximize(n_iter=num_of_iteration)
    pd.DataFrame(rf_bo.max).to_csv(BO_root+get_related_path(material_ID),mode="a")

def run_Grid_search(num_of_iteration):
    print(num_of_iteration)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    stratigy="BO"

    parser = argparse.ArgumentParser(description='bayes optimazation(--BO) or grid_search(--GS) ')
    parser.add_argument('--BO', type=int, help='Bayes optimize with number of iteration')
    parser.add_argument('--GS', type=bool, help='type use')

    args = parser.parse_args()

    if args.BO is not None or stratigy=="BO":
        for i in range(rank,128,size):
            run_bayes_optimize(args.BO,i)
    elif args.GS is not None or stratigy=="GS":
        run_Grid_search(args.GS)


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

