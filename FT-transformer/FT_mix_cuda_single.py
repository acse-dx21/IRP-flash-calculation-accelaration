import sys

sys.path.append("..")
from data import generate_data
import pandas as pd
from mpi4py import MPI
import itertools
from sklearn.model_selection import GridSearchCV
import time
import os
from model import ArtificialNN
from pytorch_tabnet.tab_model import TabNetRegressor
import torch.optim as optim
import rtdl
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import zero
data_set_index = [0,1,2,3]
mix_index = "all"
device = "cuda"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_data = True
from torch.optim.lr_scheduler import ReduceLROnPlateau
task_type = 'regression'

def get_related_path(Material_ID):
    print(Material_ID)
    print(type(Material_ID))
    if isinstance(Material_ID, list):
        return "mix_" + str(len(Material_ID[0])) + os.sep + "mixture.csv"
    else:
        return "mix_" + str(len(Material_ID)) + os.sep + str(Material_ID) + ".csv"
    print("func:get_related_path  problem")
    raise RuntimeError



# @Misc{,
#     author = {Fernando Nogueira},
#     title = {{Bayesian Optimization}: Open source constrained global optimization tool for {Python}},
#     year = {2014--},
#     url = " https://github.com/fmfn/BayesianOptimization"
# }
# import os
#
# data_path = ".." + os.sep + "cleaned_data" + os.sep
# result_path = "." + os.sep + "result_data" + os.sep
#
#

from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error

#

data_record = { "test_time_consume(s)": []}

import argparse
# print(a)
from mpi4py import MPI


def model_cv(**kwargs):
    start=time.time()
    n_block = kwargs["n_blocks"]

    ffn_d_hidden = kwargs["ffn_d_hidden"]
    attention_dropout = kwargs["attention_dropout"]  # 0.2,
    ffn_dropout = kwargs["ffn_dropout"]  # 0.1,
    residual_dropout = kwargs["residual_dropout"]

    X = {}
    y = {}
    max_epochs = 1000
    patience=100
    X['train'], X['val'], y['train'], y['val'] = X_train, X_val, y_train, y_val
    X['test'],y['test']= X_test,y_test

    print("train_shape",X['train'].shape,"val_shape",X['val'].shape)
    checkpoint_path = model_save_path + get_related_path(Material_ID).replace(".csv", ".pt")
    # not the best way to preprocess features, but enough for the demonstration

    X = {
        k: torch.tensor(v, device=device)
        for k, v in X.items()
    }
    y = {k: torch.tensor(v, device=device) for k, v in y.items()}

    # !!! CRUCIAL for neural networks when solving regression problems !!!
    # if task_type == 'regression':
    #     y_mean = y['train'].mean().item()
    #     y_std = y['train'].std().item()
    #     y = {k: (v - y_mean) / y_std for k, v in y.items()}
    # else:
    #     y_std = y_mean = None

    if task_type != 'multiclass':
        y = {k: v.float() for k, v in y.items()}


    model =rtdl.FTTransformer.make_baseline(
        n_num_features=X_train.shape[1],
        cat_cardinalities=None,
        last_layer_query_idx=[-1],
        d_token=192,  # 192,
        n_blocks=int(n_block),  # 3,
        ffn_d_hidden=int(ffn_d_hidden),
        attention_dropout=attention_dropout,  # 0.2,
        ffn_dropout=ffn_dropout,  # 0.1,
        residual_dropout=residual_dropout,
        d_out=y_train.shape[1])
    model.to(device)
    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, rtdl.FTTransformer)
        else torch.optim.AdamW(model.parameters(), lr=0.001,weight_decay=0)
    )
    loss_fn = (
        F.binary_cross_entropy_with_logits
        if task_type == 'binclass'
        else F.cross_entropy
        if task_type == 'multiclass'
        else F.mse_loss
    )

    def apply_model(x_num, x_cat=None):
        if isinstance(model, rtdl.FTTransformer):
            return model(x_num, x_cat)
        elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
            assert x_cat is None
            return model(x_num)
        else:
            raise NotImplementedError(
                f'Looks like you are using a custom model: {type(model)}.'
                ' Then you have to implement this branch first.'
            )

    @torch.no_grad()
    def evaluate(X, y):

        model.eval()
        prediction = []
        for batch in zero.iter_batches(X, 1024):
            # batch.to(device)
            prediction.append(model(batch, None))
        prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
        target = y.cpu().numpy()

        score = sklearn.metrics.mean_squared_error(target, prediction)

        return score

    batch_size = 1024
    train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)

    # Create a progress tracker for early stopping
    # Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html

    progress = zero.ProgressTracker(patience=patience)
    start_train = time.time()

    report_frequency = len(X['train']) // batch_size // 5
    start_train = time.time()

    for epoch in range(1, max_epochs + 1):
        for iteration, batch_idx in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            x_batch = X['train'][batch_idx]
            y_batch = y['train'][batch_idx]
            loss = loss_fn(apply_model(x_batch).squeeze(1), y_batch)
            loss.backward()
            optimizer.step()
            if iteration % report_frequency == 0:
                print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

        val_score = evaluate(X['val'], y['val'])

        if epoch % 10 == 0:
            test_score = evaluate(X['test'], y['test'])
            print(
                f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f} | {time.time() - start:.1f}s',
            )
        progress.update(-val_score)
        if progress.success:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss, },
                       checkpoint_path)

        if progress.fail or epoch == max_epochs:

            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            val_score = evaluate(X['val'], y['val'])

            start_test=time.time()
            test_score = evaluate(X['test'], y['test'])

            print('Checkpoint recovered:')
            print(
                f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f} | {time.time() - start:.1f}s',
                end='')
            break
    for i in range(100):
        start_pred = time.time()
        pred = model([X['test'][i, ...]],None)
        test_time = time.time() - start_pred

        data_record["test_time_consume(s)"].append(test_time)
    print("test time",test_time)



    data_record["test_time_consume(s)"].append(test_time)

    return -1

from sklearn.model_selection import train_test_split

def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train, X_val, y_val, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]
    preprocess = sklearn.preprocessing.StandardScaler().fit(X_train)

    X_train = preprocess.transform(X_train)
    print(X_train)
    X_test = preprocess.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=4)

    rf_bo = BayesianOptimization(
        model_cv,
        {
            "d_token": [128, 256],
            "n_blocks": [2, 5],
            "ffn_d_hidden": [200, 400],
            "attention_dropout": [0.1, 0.4],  # 0.2,
            "ffn_dropout": [0.01, 0.2],  # 0.1,
            "residual_dropout": [0.00, 0.05]
        }
    )

    rf_bo.maximize(init_points=5,n_iter=num_of_iteration)

    #save data
    result_data_root = "." + os.sep + "BO_result_data" + os.sep
    routing_data_root = "." + os.sep + "BO_training_routing" + os.sep

    if save_data:
        if os.path.exists(saved_root + result_data_root + get_related_path(Material_ID)):
            pd.concat([pd.DataFrame(rf_bo.res),pd.read_csv(saved_root + result_data_root + get_related_path(Material_ID),index_col=0,comment="#")],ignore_index=True).to_csv(saved_root + result_data_root + get_related_path(Material_ID))
            f = open(saved_root + result_data_root + get_related_path(Material_ID), "a+")
            f.write(f"# {device}")
            f.close()
            pd.concat([pd.DataFrame(data_record),
                       pd.read_csv(saved_root + routing_data_root + get_related_path(Material_ID), index_col=0,
                                   comment="#")], ignore_index=True).to_csv(
                saved_root + routing_data_root + get_related_path(Material_ID))
            f = open(saved_root + routing_data_root + get_related_path(Material_ID), "a+")
            f.write(f"# {device}")
            f.close()
        else:
            pd.DataFrame(rf_bo.res).to_csv(saved_root + result_data_root + get_related_path(Material_ID))
            f = open(saved_root + result_data_root + get_related_path(Material_ID), "a+")
            f.write(f"# {device}")
            f.close()
            pd.DataFrame(data_record).to_csv(saved_root + routing_data_root + get_related_path(Material_ID))
            f = open(saved_root + routing_data_root + get_related_path(Material_ID), "a+")
            f.write(f"# {device}")
            f.close()
            data_record["test_time_consume(s)"].clear()

def run_Grid_search(num_of_iteration):
    print(num_of_iteration)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for index in range(rank, len(data_set_index), size):
        data_index = data_set_index[index]
        print(data_index)

        model_save_path = "." + os.sep + "saved_model" + os.sep + "mini_dataset_mixture_"+device + os.sep + f"mini_data_{data_index}" + os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root + f"mini_data_{data_index}" + os.sep
        saved_root = "." + os.sep + "mini_cleaned_data_mixture_"+device + os.sep + f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(5, 2)