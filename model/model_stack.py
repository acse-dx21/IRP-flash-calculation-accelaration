import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
class stacking_model:
    def __init__(self,name,kfold=5,strategy="res"):
        self.name=name
        self.models=[]
        self.kfold=KFold(n_splits=kfold)
        self.strategy=strategy
        self.data_record={"trainning_time_consume(s)":{},"train_loss_record":{}}

    def add_model(self,model):
        self.models.append(model)


    def fit(self,train,target):
        shape=target.shape

        for model in self.models:
            train_temp=[]
            start = time.time()
            for train_indx, val_indx in self.kfold.split(train):

                t_x, val_x = train[train_indx], train[val_indx]

                t_y,tar_y=target[train_indx],target[train_indx]
                try:
                    model.fit(t_x,t_y)
                except:
                    print(model,"is not able to '.fit(t_x,t_y)' ")
                    raise RuntimeError

                try:
                    train_temp.append(model.predict(val_x))
                except:
                    print(model,"is not able to '.predict(val_x)' ")
                    raise RuntimeError

            self.data_record["trainning_time_consume(s)"][str(type(model))] = time.time() - start
            self.data_record["train_loss_record"][str(type(model))] = mean_squared_error(model.predict(train), target)

            if self.strategy == "res":
                target = target - model.predict(train)
                self.data_record["train_loss_record"][str(type(model))] = mean_squared_error(target, np.zeros(target.shape))

            train = np.concatenate(train_temp)


    def predict(self, x):
        if self.strategy=="res":
            result = self.models[0].predict(x)
            for i in range(1,len(self.models)):
                result += self.models[i].predict(result)
            return result

        else:
            for model in self.models:
                x=model.predict(x)
            return x

    def score(self,x,y):
        if self.strategy == "res":
            result = self.models[0].predict(x)
            for i in range(1, len(self.models)):
                result += self.models[i].predict(result)
            return -mean_squared_error(result, y)
        else:
            for model in self.models:
                x=model.predict(x)
            return -mean_squared_error(x,y)


from torch.utils.data import TensorDataset,DataLoader
import torch
# modls=stacking_model("test")
# print()
num = 200
data = np.ones((num, 14)) + np.random.randn(num, 14) * 10
target = np.ones((num, 10)) + np.random.randn(num, 10) * 0.1


import xgboost
import lightgbm
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import ArtificialNN
stak_model=stacking_model("test")
model1=xgboost.XGBRegressor()
model2=RandomForestRegressor()
model3=ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.ANN,{"input_num":10,"output_num":10})


stak_model.add_model(model1)
stak_model.add_model(model2)
stak_model.add_model(model3)
stak_model.fit(data,target)
print(stak_model.score(data,target))
print(stak_model.predict(data))
print(target)
print(stak_model.data_record)


