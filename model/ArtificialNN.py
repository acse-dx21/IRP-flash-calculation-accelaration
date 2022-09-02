import torch
import time
import torch.nn as nn
import sys
sys.path.append("..")
import numpy as np
import sklearn.metrics as metrics
import zero
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


from torch.utils.data import TensorDataset, DataLoader


class FeatureBlock(nn.Module):
    def glu(self, x, devide):
        """Generalized linear unit nonlinear activation."""
        return x[:, :devide] * torch.sigmoid()(x[:, devide:])

    def __init__(self, input_dim, feature_dim, apply_glu=False, bn_momentum=0.9, fc=None, epsilon=1e-5):
        super(FeatureBlock, self).__init__()
        self.apply_gpu = apply_glu
        self.feature_dim = feature_dim
        units = feature_dim * 2 if apply_glu else feature_dim
        self.fc = nn.Linear(input_dim, units, bias=False) if fc is None else fc
        self.bn = nn.BatchNorm1d(units, momentum=bn_momentum)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        if self.apply_gpu:
            return self.glu(x, self.feature_dim)  # GLU activation applied to BN output

        return x


class FeatureTransformer(nn.Module):
    def __init__(
            self,
            input_dim,
            feature_dim,

            share_fcs=[],
            n_shared=2,
            n_total=4,
            bn_momentum=0.9,
    ):
        super(FeatureTransformer, self).__init__()
        self.n_total, self.n_shared = n_total, n_shared
        kwrgs = {
            "input_dim": feature_dim,
            "feature_dim": feature_dim,
            "bn_momentum": bn_momentum,
        }

        kwrgs0 = {
            "input_dim": input_dim,
            "feature_dim": feature_dim,
            "bn_momentum": bn_momentum,
        }
        self.blocks = nn.ModuleList()
        if (len(share_fcs) > 0):
            self.blocks.append(FeatureBlock(**kwrgs0, fc=share_fcs[0]))
        else:
            self.blocks.append(FeatureBlock(**kwrgs0))

        for n in range(1, n_total):
            if share_fcs and n < len(share_fcs):
                self.blocks.append(
                    FeatureBlock(**kwrgs, fc=share_fcs[n]))  # Building shared blocks by providing FC layers
            else:
                self.blocks.append(FeatureBlock(**kwrgs))

    def forward(self, x):
        x = self.blocks[0](x)
        for n in range(1, self.n_total):
            x = x * np.sqrt(0.5) + self.blocks[n](x)
        return x

    @property
    def shared_fcs(self):
        return [self.blocks[i].fc for i in range(self.n_shared)]


from sparsemax import Sparsemax


class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(AttentiveTransformer, self).__init__()
        self.block = FeatureBlock(
            input_dim,
            feature_dim,
            apply_glu=False,  # sparsemax instead of glu
        )

    def forward(self, x, prior_scales):
        x = self.block(x)

        return Sparsemax()(x * prior_scales)


# x=torch.randn((10,100))
# a=FeatureTransformer(100,100)
# print(a(x).shape)

def sparse_loss(at_mask):
    # print(-at_mask,torch.log(at_mask + 1e-15))
    # print(torch.multiply(-at_mask,  torch.log(at_mask + 1e-15)))
    # print(torch.sum( torch.multiply(-at_mask,  torch.log(at_mask + 1e-15)),
    #                   axis=1))

    loss = torch.mean(
        torch.sum(torch.multiply(-at_mask, torch.log(at_mask + 1e-15)),
                  axis=1)
    )

    return loss



class my_tabnet(nn.Module):
    def __init__(self,
                 input_dim,
                 inner_feature_dim,
                 out_feature_dim,
                 output_dim,
                 n_step=2,
                 n_total=4,
                 n_shared=2,
                 relaxation_factor=1.5,
                 bn_epsilon=1e-5,
                 bn_momentum=0.7,
                 sparsity_coefficient=1e-5):
        super(my_tabnet, self).__init__()
        self.output_dim, self.input_dim = output_dim, input_dim

        self.inner_feature_dim, self.out_feature_dim = inner_feature_dim, out_feature_dim

        self.bn = nn.BatchNorm1d(input_dim, eps=bn_epsilon, momentum=bn_momentum)

        self.relaxation_factor = torch.tensor(relaxation_factor, requires_grad=False)

        self.n_step = n_step

        kargs = {
            "input_dim": input_dim,
            "feature_dim": inner_feature_dim + out_feature_dim,
            "n_total": n_total,
            "n_shared": n_shared,
            "bn_momentum": bn_momentum
        }
        self.feature_transforms = nn.ModuleList()
        self.attentive_transforms = nn.ModuleList()
        self.feature_transforms.append(FeatureTransformer(**kargs))

        for i in range(n_step):
            self.feature_transforms.append(
                FeatureTransformer(**kargs, share_fcs=self.feature_transforms[0].shared_fcs)
            )
            self.attentive_transforms.append(
                AttentiveTransformer(inner_feature_dim, input_dim)
            )

        self.act = nn.ReLU()
        self.output = nn.Linear(out_feature_dim, output_dim)

    def forward(self, features):
        bs = features.shape[0]
        features = self.bn(features)  # Batch Normalisation
        masked_features = features
        out_agg = torch.zeros((bs, self.out_feature_dim), requires_grad=False).to(features.device)

        prior_scales = torch.ones((bs, self.input_dim)).to(features.device)
        for step_i in range(self.n_step + 1):
            inner_feature = self.feature_transforms[step_i](
                masked_features
            )
            if (step_i > 0):
                out = self.act(inner_feature[..., : self.out_feature_dim])
                # print(out.shape)
                out_agg += out


            if step_i < self.n_step:
                feature_for_mask = inner_feature[:, self.out_feature_dim:]
                mask_values = self.attentive_transforms[step_i](
                    feature_for_mask, prior_scales
                )
                # print("here",mask_values.shape)
                prior_scales = prior_scales * self.relaxation_factor - mask_values
                masked_features = torch.multiply(mask_values, features)

        final_output = self.output(out)
        return final_output

class continues_pred:
    def __init__(self,model):
        self.model=model
        self.continues_predict_data={}


    def continues_predict(self, data, Ts, Ps):
        criterian=metrics.mean_squared_error
        data.set_return_type("test")
        X_test, y_test, material_IDs = data[2]
        out = self.model.predict(X_test)
        print("first_loss",criterian(out,y_test))
        losses=[]
        cnt=0
        for T, P in zip(Ts, Ps):
            # accumuted error
            X_test[..., -2:] = self.compute_material_fraction(out,len(material_IDs[0]))
            X_test[..., -3] = P
            X_test[..., -4] = T
            out = self.model.predict(X_test)

            # real_solution
            data.set_Ps(P)
            data.set_Ts(T)
            X_test_real, y_test_real, material_IDs_real = data[2]

            print(cnt,"loss:",criterian(out,y_test_real.astype(np.float32)))
            cnt+=1
            losses.append(criterian(out,y_test_real.astype(np.float32)))

        self.continues_predict_data["continues_predict_loss"]=losses
        self.continues_predict_data["Ps"]=Ps
        self.continues_predict_data["Ts"] = Ts
        print(self.continues_predict_data)
        return out, X_test

    def compute_material_fraction(self, out,material_num):

        return (out[..., :material_num].T * out[..., -2] + out[..., material_num:2 * material_num].T * out[..., -1]).T


class Neural_Model_Sklearn_style():
    def __init__(self, core, core_parameter):
        print(core.__name__)
        self.model = core(**core_parameter)
        self.data_record = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.continues_predict_data = {}

    def fit(self, train, target, max_epochs=100, batch_size=128, eval_set=None, patience=10, verbose=10, my_criterion=None, optimizer=None,mission=0):
        if my_criterion:
            criterion = my_criterion
        else:
            criterion=nn.MSELoss()

        if eval_set:
            assert isinstance(eval_set, list)
            assert isinstance(eval_set[0], tuple)

        checkpoint_path=f"temp_model{mission}.pt"
        progress = zero.ProgressTracker(patience=patience)
        self.model.to(self.device)
        self.model.train()
        print(self.device)
        Data_loader = DataLoader(
            TensorDataset(torch.from_numpy(train.astype(np.float32)), torch.from_numpy(target.astype(np.float32))),
            batch_size=batch_size, shuffle=True)
        if optimizer == None:
            optimizer = torch.optim.Adam(self.model.parameters())
            optimizer.zero_grad()
        train_loss_record = []
        train_time_record = []
        val_loss_record = {}
        start = time.time()
        cnt=0
        for epoch in range(max_epochs):
            loss_to_mean=0
            for x, y in Data_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                optimizer.zero_grad()
                # if isinstance(criterion, loss_function.My_Mass_Balance_loss):
                #     loss = criterion(y_pred, y, x[:, -self.model.material_num:])
                # else:
                loss = criterion(y_pred, y)
                loss.backward()
                loss_to_mean+=loss.item()*y.shape[0]
                optimizer.step()
            loss_eval=0
            if eval_set:
                for eval_loader in eval_set:
                    eval_cnt=0

                    x_eval, y_eval = eval_loader
                    loss_eval=self.score(x_eval, y_eval)
                    val_loss_record[eval_cnt]=loss_eval
                    eval_cnt+=1


            train_loss_record.append(loss_to_mean/target.shape[0])
            train_time_record.append(time.time() - start)
            if not epoch % verbose:
                print("epoch ",epoch," loss",train_loss_record[-1],"|",end=" " )
                if eval_set:
                        evl_result=[]
                        for key,itm in val_loss_record.items():
                            print(f"val_{key}_mse", itm,end=" ")
                            evl_result.append(itm)
                print()

            if eval_set:
                progress.update(-np.mean(evl_result))
                if progress.success:
                    torch.save({'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss_record[-1], },
                               checkpoint_path)

                if progress.fail or epoch == max_epochs:

                    checkpoint = torch.load(checkpoint_path)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epoch = checkpoint['epoch']
                    loss = checkpoint['loss']
                    val_score = self.score(x_eval, y_eval)

                    print('Checkpoint recovered:')
                    print(
                        f'Epoch {epoch:03d} | Validation score: {val_score:.4f} |  {time.time() - start:.1f}s',
                        end='')
                    break
        # record each epoch
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
        test_loss_record = 0
        for x, y in Test_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            # loss = self.criterion(y_pred, y,x[:,-self.material_num:])
            loss = criterion(y_pred, y)
            test_loss_record+=loss.cpu().item()*y.shape[0]

        loss = test_loss_record/train.shape[0]
        # one_batch(128)
        self.data_record["test_loss_record"] = loss
        self.data_record["test_time_consume(s)"] = time.time() - start

        return loss

    def __str__(self):
        return "Model prepared on " + self.device

    def predict(self, x):
        self.model.eval()
        return self.model(torch.tensor(x.astype(np.float32)).to(self.device)).detach().cpu().numpy()

    def continues_predict(self, data, Ts, Ps):
        criterian=nn.MSELoss()
        data.set_return_type("test")
        X_test, y_test, material_IDs = data[2]
        X_test, y_test,self.model= torch.tensor(X_test.astype(np.float32)).to(self.device), torch.tensor(y_test.astype(np.float32)).to(self.device),self.model.to(self.device)
        self.model.eval()
        out = self.model(X_test)
        print("first_loss",criterian(out,y_test))
        losses=[]

        for T, P in zip(Ts, Ps):
            # accumuted error
            X_test[..., -2:] = self.compute_material_fraction(out,len(material_IDs[0]))
            X_test[..., -3] = P
            X_test[..., -4] = T
            out = self.model(X_test)

            # real_solution
            data.set_Ps(P)
            data.set_Ts(T)
            X_test_real, y_test_real, material_IDs_real = data[2]

            print("loss:",criterian(out,torch.tensor(y_test_real.astype(np.float32)).to(self.device)))
            losses.append(criterian(out,torch.tensor(y_test_real.astype(np.float32)).to(self.device)).cpu().item())

        self.continues_predict_data["continues_predict_loss"]=losses
        self.continues_predict_data["Ps"]=Ps
        self.continues_predict_data["Ts"] = Ts
        print(self.continues_predict_data)
        return out.detach().cpu().numpy(), X_test

    def compute_material_fraction(self, out,material_num):

        return (out[..., :material_num].T * out[..., -2] + out[..., material_num:2 * material_num].T * out[..., -1]).T

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

    def save_model(self,path):
        torch.save(self.model.state_dict(), path)

    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
