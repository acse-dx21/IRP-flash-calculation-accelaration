import sys

sys.path.append("..")
import os
from data import generate_data
import tool
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from data.generate_data import genertate_T, genertate_P, genertate_Zs_n

from my_test import test_data_set
from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, SRKMIX, FlashVLN, PropertyCorrelationsPackage, \
    HeatCapacityGas
from torch.utils.data import DataLoader
from model import ArtificialNN
from model.train import DeepNetwork_Train

import pandas as pd
import numpy as np
import itertools

from itertools import combinations
from scipy.special import comb, perm
from sklearn.model_selection import GridSearchCV
import time
from pytorch_tabnet.tab_model import TabNetRegressor
import torch.optim as optim

data_set_index = [2,]
mix_index="all"
device = "cuda"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_data=False
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




X_train=0
y_train=0
X_test=0
y_test=0
Material_ID = 0

param_grid = [
    {'subsample': [0.2, 0.6, 1.0],
     'learning_rate': [0.01, 0.05, 0.1],
     'n_estimators': [300, 400, 500],
     'max_depth': [3, 5, 10],
     'colsample_bytree': [0.6],
     'reg_lambda': [10]}
]




from model.train import check_IDexist
from tool.log import log

import os

data_path = ".." + os.sep +"data"+os.sep+"mini_cleaned_data" + os.sep
result_path = "." + os.sep + "MSELoss_data" + os.sep


from mpi4py import MPI



# model=ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.simple_ANN,{"material":Material_ID})
# model.fit(Data_loader=train_loader,epoch=10)
# print(model.get_data)
# model.score(test_loader)
# print(model.get_data)

def get_related_path(Material_ID):
    if isinstance(Material_ID,list):
        return "mix_"+str(len(Material_ID[0]))+os.sep+"mixture.csv"
    return "mix_"+str(len(Material_ID))+os.sep+str(Material_ID)+".csv"

data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}


def compute_material_fraction(out, material_num):
    return (out[..., :material_num].T * out[..., -2] + out[..., material_num:2 * material_num].T * out[..., -1]).T

from sklearn.model_selection import KFold
import numpy as np


def model_cv():

    epochs=1000
    model_instance = TabNetRegressor()
    model_save_path="."+os.sep+"saved_model"+os.sep+"mini_dataset_mixture_cuda"+os.sep+"mini_data_2"+ os.sep
    model_instance.load_model(model_save_path + get_related_path(Material_ID).replace("mixture","mixture_havepre").replace(".csv", ".zip"))
    loss_data={"target":[]}

    cnt=0
    print("train_shape", X_train.shape, "X_test_shape", X_test.shape)
    if True:
        conti_data_saved_root = saved_root + "continues_pred" + os.sep + "mix_2" + os.sep + f"mixture_experience{cnt}.csv"
        X_test_conti = np.copy(preprocess.inverse_transform(X_test))
        y_test_conti = np.copy(y_test)  # in order to make code look nice, we copy it too.
        try:
            for i in range(epochs):
                pred = model_instance.predict(preprocess.transform(X_test_conti))
                loss = mean_squared_error(pred, y_test_conti)
                loss_data["target"].append(loss)
                X_test_conti[..., -2:] = compute_material_fraction(pred, 2)
                print(i, loss)
                print("----------------X----------------", )
                print(X_test_conti)
                print("---------------------------------", )
        except:
            pass
        finally:
            pd.DataFrame(loss_data).to_csv(conti_data_saved_root)
        cnt += 1

    exit()
    return -mean_squared_error(pred, y_test)

import argparse
# print(a)
from mpi4py import MPI

import sklearn
from sklearn.model_selection import train_test_split

def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train, X_val, y_val, X_test, y_test, Material_ID,preprocess
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]
    preprocess = sklearn.preprocessing.StandardScaler().fit(X_train)

    X_train = preprocess.transform(X_train)

    X_test = preprocess.transform(X_test)
    print(X_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=4)

    print(model_save_path + get_related_path(Material_ID).replace(".csv", ".json"))
    model_cv()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    start,end = get_range(mix_index)


    for index in range(rank, len(data_set_index), size):
        data_index = data_set_index[index]
        print(data_index)

        model_save_path = "." + os.sep + "saved_model" + os.sep + "mini_dataset_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root + f"mini_data_{data_index}" + os.sep
        saved_root = "." + os.sep + "mini_cleaned_data_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(1, 2)


