import torch
import time
import torch.nn as nn
from torch.autograd.function import Function as Function
import numpy as np


class ANN(nn.Module):
    name = "neural_network"

    def __init__(self, input_num, output_num, deepth=4, Nodes_per_layer=300):
        super(ANN, self).__init__()

        self.input = nn.Linear(input_num, Nodes_per_layer)
        self.drop = nn.Dropout(0.25)
        self.BN = nn.BatchNorm1d(Nodes_per_layer)
        self.layers = nn.ModuleList()
        for i in range(1, deepth):
            self.layers.append(nn.Linear(Nodes_per_layer, Nodes_per_layer))
        self.output = nn.Linear(self.layers[-1].out_features, output_num)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.input(x))
        for layer in self.layers:
            x = self.drop(self.BN(self.act(layer(x))))
        x = self.act(self.output(x))
        return x


class simple_ANN(nn.Module):
    name = "neural_network"

    def __init__(self, material, deepth=3, Nodes_per_layer=100):
        super(simple_ANN, self).__init__()
        self.material = material
        self.material_num = len(material)
        self.input = nn.Linear(self.material_num * 4 + 2, Nodes_per_layer)
        self.drop = nn.Dropout(0.25)
        self.BN = nn.BatchNorm1d(Nodes_per_layer)
        self.layers = nn.ModuleList()
        for i in range(1, deepth):
            self.layers.append(nn.Linear(Nodes_per_layer, Nodes_per_layer))
        self.output = nn.Linear(self.layers[-1].out_features, self.material_num * 2 + 2)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.input(x))
        for layer in self.layers:
            x = self.drop(self.BN(self.act(layer(x))))
        x = self.act(self.output(x))
        return x

class ANN_material0(nn.Module):
    name = "neural_network"

    def __init__(self, material, deepth=3, Nodes_per_layer=100):
        super(ANN_material0, self).__init__()
        self.material = material
        self.material_num = len(material)
        self.input = nn.Linear(self.material_num * 4 + 2, Nodes_per_layer)
        self.drop = nn.Dropout(0.25)
        self.BN = nn.BatchNorm1d(Nodes_per_layer)
        self.layers = nn.ModuleList()
        for i in range(1, deepth):
            self.layers.append(nn.Linear(Nodes_per_layer, Nodes_per_layer))
        self.output = nn.Linear(self.layers[-1].out_features, 2)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.input(x))
        for layer in self.layers:
            x = self.drop(self.BN(self.act(layer(x))))
        x = self.act(self.output(x))
        return x


class My_MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))


class Mass_Balence_loss(nn.Module):
    def __init__(self, Beta=None):
        self.Beta = Beta if Beta is not None else 1

    def forward(self, M1, M2):
        return torch.abs(M1 - M2)


class Neural_Model_Sklearn_style:
    def __init__(self, core, core_parameter):

        self.model = core(**core_parameter)
        self.data_record = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, Data_loader, epoch=15, criterion=nn.MSELoss(), optimizer=None):
        self.model.train()

        if optimizer == None:
            optimizer = torch.optim.Adam(self.model.parameters())
        train_loss_record = []
        start = time.time()

        for i in range(epoch):
            for x, y in Data_loader:
                loss_to_mean = []
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

    def score(self, Test_loader, criterion=nn.MSELoss()):
        self.model.eval()
        start = time.time()
        test_loss_record = []
        for x, y in Test_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            # loss = self.criterion(y_pred, y,x[:,-self.material_num:])
            loss = criterion(y_pred, y)
            test_loss_record.append(loss.item())

        loss = np.mean(test_loss_record)

        self.data_record["test_loss_record"] = loss
        self.data_record["test_time_consume(s)"] = time.time() - start
        return loss

    def predict(self, x):
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

    def set_device(self, device):
        self.device = device
import torch.nn.functional as F
class OneD_CNN(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(OneD_CNN, self).__init__()
        cha_1 = 256
        cha_2 = 512
        cha_3 = 512

        cha_1_reshape = int(hidden_size / cha_1)
        cha_po_1 = int(hidden_size / cha_1 / 2)
        cha_po_2 = int(hidden_size / cha_1 / 2 / 2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1, cha_2, kernel_size=5, stride=1, padding=2, bias=False),
                                          dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True),
                                          dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True),
                                            dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_3, kernel_size=5, stride=1, padding=2, bias=True),
                                            dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0], self.cha_1,
                      self.cha_1_reshape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x = x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


from torch.utils.data import TensorDataset, DataLoader


class Neural_Model_Sklearn_style:
    def __init__(self, core, core_parameter):
        print(core.__name__)
        self.model = core(**core_parameter)
        self.data_record = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train, target, epoch=15, batch_size=128, criterion=nn.MSELoss(), optimizer=None):
        self.model.to(self.device)
        self.model.train()
        Data_loader = DataLoader(
            TensorDataset(torch.from_numpy(train.astype(np.float32)), torch.from_numpy(target.astype(np.float32))),
            batch_size=batch_size, shuffle=True)
        if optimizer == None:
            optimizer = torch.optim.Adam(self.model.parameters())
        train_loss_record = []
        train_time_record = []
        start = time.time()

        for i in range(epoch):
            for x, y in Data_loader:
                loss_to_mean = []
                time_record = []
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
        #record each epoch
        self.data_record["trainning_time_consume(s)"] = train_time_record
        self.data_record["train_loss_record"] = train_loss_record
        return self.data_record

    def score(self, train, target, criterion=nn.MSELoss(), batch_size=128):
        self.model.to(self.device)
        self.model.eval()

        Test_loader = DataLoader(
            TensorDataset(torch.from_numpy(train.astype(np.float32)), torch.from_numpy(target.astype(np.float32))),
            batch_size=batch_size, shuffle=True)
        start = time.time()
        test_time_record = []
        test_loss_record = []
        for x, y in Test_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            # loss = self.criterion(y_pred, y,x[:,-self.material_num:])
            loss = criterion(y_pred, y)
            test_loss_record.append(loss.cpu().item())

        loss = np.mean(test_loss_record)
        #one_batch(128)
        self.data_record["test_loss_record"] = loss
        self.data_record["test_time_consume(s)"] = time.time() - start

        return loss

    def __str__(self):
        return "Model prepared on "+ self.device

    def predict(self, x):
        return self.model(torch.tensor(x.astype(np.float32)).to(self.device)).detach().cpu().numpy()

    @property
    def get_device(self):
        return self.device

    @property
    def get_core_name(self):
        return type(self.model).__name__

    @property
    def get_data(self):
        return self.data_record

    def set_device(self, device):
        self.device = device






