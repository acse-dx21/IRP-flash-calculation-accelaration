import generate_data
from data.generate_data import genertate_T, genertate_P, genertate_Zs_n,flashdata
# from tool import Py2Cpp as pc
from my_test import test_data_set
import xgboost as xgb
from xgboost import XGBRegressor,XGBClassifier
from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, SRKMIX, FlashVLN, PropertyCorrelationsPackage, \
    HeatCapacityGas
from torch.utils.data import DataLoader
from model import ArtificialNN
from model.train import DeepNetwork_Train ,check_IDexist
import torch
import torch.nn as nn
import thermo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import convolve2d, correlate2d
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
import torch
from mpi4py import MPI
import random
import os

import csv

data_set_index = [3]
mix_index = "all"
device = "cuda"
data_root = "." + os.sep + "mini_cleaned_data2" + os.sep
#
if __name__ == "__main2__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for index in range(rank, len(data_set_index), size):
        data_index = data_set_index[index]
        print(data_index)

        model_save_path = "." + os.sep + "saved_model" + os.sep + "mini_dataset_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root + f"mini_data_{data_index}" + os.sep
        saved_root = "." + os.sep + "mini_cleaned_data_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        print(mini_data_path, os.listdir(mini_data_path))
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        X_train, y_train, X_test, y_test, Material_ID = relate_data[2]
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,Material_ID,len(Material_ID))
        assert Material_ID==[('Ethane', 'Heptane'), ('Ethane', 'N-Butane'), ('Ethane', 'N-Hexane'), ('Ethane', 'N-Pentane'), ('Ethane', 'Propane'), ('Methane', 'Ethane'), ('Methane', 'Heptane'), ('Methane', 'N-Butane'), ('Methane', 'N-Hexane'), ('Methane', 'N-Pentane'), ('Methane', 'Propane'), ('N-Butane', 'Heptane'), ('N-Butane', 'N-Hexane'), ('N-Butane', 'N-Pentane'), ('N-Hexane', 'Heptane'), ('N-Pentane', 'Heptane'), ('N-Pentane', 'N-Hexane'), ('Propane', 'Heptane'), ('Propane', 'N-Butane'), ('Propane', 'N-Hexane'), ('Propane', 'N-Pentane')]

