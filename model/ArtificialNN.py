
import torch
import time
import torch.nn as nn
from torch.autograd.function import Function as Function
import numpy as np

class ANN(nn.Module):

    name="neural_network"
    def __init__(self,input_num,output_num,deepth=3,Nodes_per_layer=100):
        super(ANN,self).__init__()

        self.input = nn.Linear(input_num, Nodes_per_layer)
        self.drop = nn.Dropout(0.25)
        self.BN=nn.BatchNorm1d(Nodes_per_layer)
        self.layers=nn.ModuleList()
        for i in range(1,deepth):
            self.layers.append(nn.Linear(Nodes_per_layer, Nodes_per_layer))
        self.output = nn.Linear(self.layers[-1].out_features, output_num)
        self.act=nn.Sigmoid()


    def forward(self,x):
        x = self.act(self.input(x))
        for layer in self.layers:
            x= self.drop(self.BN(self.act(layer(x))))
        x = self.act(self.output(x))
        return x

class simple_ANN(nn.Module):
    name = "neural_network"
    def __init__(self,material,deepth=3,Nodes_per_layer=100):
        super(simple_ANN,self).__init__()
        self.material=material
        self.material_num=len(material)
        self.input = nn.Linear(self.material_num * 4 + 2, Nodes_per_layer)
        self.drop = nn.Dropout(0.25)
        self.BN=nn.BatchNorm1d(Nodes_per_layer)
        self.layers=nn.ModuleList()
        for i in range(1,deepth):
            self.layers.append(nn.Linear(Nodes_per_layer, Nodes_per_layer))
        self.output = nn.Linear(self.layers[-1].out_features, self.material_num*2+2)
        self.act=nn.Sigmoid()


    def forward(self,x):
        x = self.act(self.input(x))
        for layer in self.layers:
            x= self.drop(self.BN(self.act(layer(x))))
        x = self.act(self.output(x))
        return x






class My_MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))


class Mass_Balence_loss(nn.Module):
    def __init__(self,Beta=None):
        self.Beta=Beta if Beta is not None else 1

    def forward(self,M1,M2):
        return torch.abs(M1-M2)


class Neural_Model_Sklearn_style:
    def __init__(self,core,core_parameter):

        self.model=core(**core_parameter)
        self.data_record={}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, Data_loader, epoch=15, criterion=nn.MSELoss(),optimizer = None):
        self.model.train()

        if optimizer==None:
            optimizer=torch.optim.Adam(self.model.parameters())
        train_loss_record = []
        start = time.time()


        for i in range(epoch):
            for x, y in Data_loader:
                loss_to_mean=[]
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                optimizer.zero_grad()
                if not isinstance(criterion, nn.MSELoss):
                    loss = criterion(y_pred, y, x[:, -self.model.material_num:])
                else:
                    loss = criterion(y_pred, y)
                loss.backward()
                loss_to_mean.append(loss.item())
                optimizer.step()
            train_loss_record.append(np.mean(loss_to_mean))

        self.data_record["trainning_time_consume(s)"] = time.time() - start
        self.data_record["train_loss_record"] = train_loss_record
        return self.data_record


    def score(self,Test_loader,criterion=nn.MSELoss()):
        self.model.eval()
        start = time.time()
        test_loss_record = []
        for x, y in Test_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            # loss = self.criterion(y_pred, y,x[:,-self.material_num:])
            loss = criterion(y_pred, y)
            test_loss_record.append(loss.item())

        loss=np.mean(test_loss_record)

        self.data_record["test_loss_record"]=loss
        self.data_record["test_time_consume(s)"] = time.time() - start
        return loss

    def predict(self,x):
        return self.model(x)
    @property
    def get_device(self):
        return self.device

    @property
    def get_core_name(self):
        return type(self.model).__name__

    @property
    def get_data(self):
        return self.data_record


    def set_device(self,device):
        self.device=device

from torch.utils.data import TensorDataset,DataLoader

class Neural_Model_Sklearn_style:
    def __init__(self,core,core_parameter):

        self.model=core(**core_parameter)
        self.data_record={}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train,target, epoch=15, batch_size=128,criterion=nn.MSELoss(),optimizer = None):
        self.model.train()
        Data_loader=DataLoader(TensorDataset(torch.from_numpy(train.astype(np.float32)),torch.from_numpy(target.astype(np.float32))),batch_size=batch_size, shuffle=True)
        if optimizer==None:
            optimizer=torch.optim.Adam(self.model.parameters())
        train_loss_record = []
        train_time_record  =[]
        start = time.time()


        for i in range(epoch):
            for x, y in Data_loader:
                loss_to_mean=[]
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                optimizer.zero_grad()
                if not isinstance(criterion, nn.MSELoss):
                    loss = criterion(y_pred, y, x[:, -self.model.material_num:])
                else:
                    loss = criterion(y_pred, y)
                loss.backward()
                loss_to_mean.append(loss.item())
                optimizer.step()
            train_loss_record.append(np.mean(loss_to_mean))
            train_time_record.append(time.time() - start)
        self.data_record["trainning_time_consume(s)"] = train_time_record
        self.data_record["train_loss_record"] = train_loss_record
        return self.data_record


    def score(self,train, target,criterion=nn.MSELoss(),batch_size=128):
        self.model.eval()

        Test_loader = DataLoader(TensorDataset(torch.from_numpy(train.astype(np.float32)), torch.from_numpy(target.astype(np.float32))), batch_size=batch_size, shuffle=True)

        start = time.time()
        test_loss_record = []
        for x, y in Test_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            # loss = self.criterion(y_pred, y,x[:,-self.material_num:])
            loss = criterion(y_pred, y)
            test_loss_record.append(loss.item())

        loss=np.mean(test_loss_record)

        self.data_record["test_loss_record"]=loss
        self.data_record["test_time_consume(s)"] = time.time() - start
        return loss

    def predict(self,x):
        return self.model(torch.tensor(x.astype(np.float32))).detach().numpy()
    @property
    def get_device(self):
        return self.device

    @property
    def get_core_name(self):
        return type(self.model).__name__

    @property
    def get_data(self):
        return self.data_record


    def set_device(self,device):
        self.device=device






