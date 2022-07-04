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
from model.loss_function import My_Mass_Balance_loss
from model import ArtificialNN
import pandas as pd


# mini_data_path=".."+os.sep+"data"+os.sep+"mini_cleaned_data"+os.sep

All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
relate_data=generate_data.multicsv_data_generater()
relate_data.set_return_type("Dataloader")
relate_data.set_batch_size(128)
train_loader=0
test_loader=0
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
    return "mix_"+str(len(Material_ID))+os.sep+str(Material_ID)+".csv"



from bayes_opt import BayesianOptimization

def model_cv(**kwargs):
    model_kwargs={}
    model_kwargs["Nodes_per_layer"] = 300
    model_kwargs["deepth"] = 3
    model_kwargs["material"]=Material_ID
    model_instance = ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.simple_ANN,model_kwargs
        )

    PINN_loss=My_Mass_Balance_loss(1,kwargs["My_Mass_Balance_loss"])
    model_instance.fit(train_loader,epoch=30,criterion=PINN_loss)

    score=model_instance.score(test_loader)

    data_root="."+os.sep+"BO_training_routing"+os.sep
    pd.DataFrame(model_instance.data_record).to_csv(data_root + get_related_path(Material_ID))

    return -score

import argparse
# print(a)
from mpi4py import MPI



def run_bayes_optimize(num_of_iteration=1,data_index=10):
    BO_root="."+os.sep+"BO_result_data"+os.sep
    global train_loader, test_loader, Material_ID
    train_loader, test_loader, Material_ID = relate_data[data_index]
    rf_bo = BayesianOptimization(
            model_cv,
        {'My_Mass_Balance_loss': [0.01, 100]}
        )

    rf_bo.maximize(n_iter=num_of_iteration)
    pd.DataFrame(rf_bo.res).to_csv(BO_root+get_related_path(Material_ID))



if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for i in range(rank, 128, size):
        run_bayes_optimize(100,i)
#
#     all_data = generate_data.multicsv_data_generater(data_path, return_type="Dataloader")
#
#     for i in range(len(all_data) - rank - 1, -1, -size):
#         Material_ID=all_data.materials[i]
#         print(Material_ID, check_IDexist(Material_ID, result_path))
#
#         if not check_IDexist(Material_ID, result_path):
#             batchsize_list = [64, 128, 512]
#             learning_rate_list = [0.001, 0.002, 0.005]
#             for batchsize in batchsize_list:
#                 for learning_rate in learning_rate_list:
#
#                     all_data.batch_size=batchsize
#                     train_loader, test_loader, Material_ID = all_data[i]
#
#                     my_model = ArtificialNN.simple_ANN(Material_ID)
#                     #         model.load_state_dict(torch.load(model_path))
#
#                     criterion = nn.MSELoss()
#                     #         criterion=loss_function.My_Mass_Balance_loss(1,1)
#                     optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
#
#
#                     ANN_train = DeepNetwork_Train(my_model, criterion, optimizer, train_loader,
#                                                   "." + os.sep + f"{type(criterion).__name__}_data" + os.sep + "mix_" + str(
#                                                       len(
#                                                           Material_ID)) + os.sep + str(Material_ID),test_loader,f"lr={learning_rate}bs={batchsize}")
#
#                     ANN_train.Train(epoch=1)
#
#                     model_paths = "." + os.sep + "saved_model" + os.sep + "simpleANN" + f"lr={learning_rate}bs={batchsize}" + type(
#                         criterion).__name__ + ".pt"
#                     torch.save(my_model.state_dict(), model_paths)
#
