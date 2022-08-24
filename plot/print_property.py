

from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, SRKMIX, FlashVLN, PropertyCorrelationsPackage, \
    HeatCapacityGas
from torch.utils.data import DataLoader

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


#
All_ID = ['Methane','Ethane', 'Propane', 'N-Butane','N-Pentane', 'N-Hexane', 'Heptane']
print("All_IDs",len(All_ID))
# ID=['Ethane']
constants, properties = ChemicalConstantsPackage.from_IDs(All_ID)
dic={"material":All_ID,"Tc":constants.Tcs,"Pc":constants.Pcs,"Ac":constants.omegas}
print(constants.Tcs, constants.Pcs, constants.omegas)

pd.DataFrame(dic).to_csv("material_property.csv")




