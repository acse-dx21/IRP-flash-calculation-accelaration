import pandas as pd
import sys
import time
sys.path.append("..")
from tool.log import epoch_log,log
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class DeepNetwork_Test:
    def __init__(self, model, criterion, dataloader, name="pure"):
        self.model = model
        self.criterion = criterion
        self.dataloader = dataloader
        self.Name = name
        self.material_num = model.material_num

        self.loss=torch.nn.MSELoss()
    def test(self):
        self.model.eval()
        for x, y in self.dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = self.model(x)
            loss = self.loss(y_pred, y)
        return loss


class DeepNetwork_Train:
    def __init__(self, model, criterion, optimizer, dataloader,file_name="pure", test_loader=None,comment="no_comment"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.Name = file_name
        self.test_loader = test_loader
        self.material=self.model.material
        self.material_num=model.material_num
        self.comment = comment

        if self.test_loader is not None:
            self.test = DeepNetwork_Test(self.model, self.criterion, self.test_loader)

    def train(self):
        self.model.train()
        for x, y in self.dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = self.model(x)
            self.optimizer.zero_grad()
            # loss = self.criterion(y_pred, y,x[:,-self.material_num:])
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
        return loss

    def Train(self, epoch, num_record=1):
        epoch_logs_train = log(self.Name + type(self.model).__name__  + "train",["epoch","loss","time_use"])
        epoch_logs_test = log(self.Name + type(self.model).__name__ + "test",["epoch","loss","time_use"])
        self.model = self.model.to(device)
        start_train=time.time()
        epoch_logs_train({"epoch": 0, "loss": self.train().item(),"time_use":time.time()-start_train})
        for i in range(1, epoch):

            if i % num_record == 0:
                log_train = {}
                log_train["epoch"] = i
                start_train = time.time()
                log_train["loss"] = self.train().item()
                log_train["time_use"]=time.time()-start_train
                epoch_logs_train(log_train)
                if self.test_loader is not None:
                    log_test = {}

                    log_test["epoch"] = i
                    start_test = time.time()
                    log_test["loss"] = self.test.test().item()
                    log_test["time_use"]=start_test-time.time()
                    epoch_logs_test(log_test)

            else:
                self.train(self.model, self.criterion, self.optimizer, self.dataloader)
        epoch_logs_train.release(str(self.model.parameters))
        epoch_logs_train.add(self.comment)
        epoch_logs_train.add(self.material)

        epoch_logs_test.release(str(self.model.parameters))
        epoch_logs_test.add(self.comment)
        epoch_logs_test.add(self.material)

    def fit(self,xs,ys):
        for x, y in zip(xs,ys):
            x, y = x.to(device), y.to(device)
            y_pred = self.model(x)
            self.optimizer.zero_grad()
            # loss = self.criterion(y_pred, y,x[:,-self.material_num:])
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()


import os
def check_IDexist(ID,rootpath):
    for root, _, fnames in os.walk(rootpath):
        try:
            for fname in fnames:
                if str(ID) in fname:
                    return True
        except:
            return False
    return False