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
data_set_index = [3]
mix_index = "all"
device = "cuda"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_model = True
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


def get_range(mix_index):
    """
    use to fine target mixture
    :param mix_index: "all" or int range from 1-7
    :return:
    """
    if (mix_index == "all"):
        return 0, 127
    assert mix_index > 0
    start = 0
    end = comb(7, 1)

    for i in range(1, mix_index):
        start += comb(7, i)
        end += comb(7, i + 1)

    return int(start), int(end)


param_grid = [
    # try combinations of hyperparameters
    {'subsample': [0.2, 0.6, 1.0],
     'learning_rate': [0.01, 0.05, 0.1],
     'n_estimators': [300, 400, 500],
     'max_depth': [3, 5, 10],
     'colsample_bytree': [0.6],
     'reg_lambda': [10]}
]


def grid_i(X_train, y_train):
    # train across 3 folds
    grid_search = GridSearchCV(TabNetRegressor(objective='reg:squarederror', n_jobs=3, random_state=42),
                               param_grid,
                               cv=3,
                               scoring='neg_mean_squared_error',
                               return_train_score=True,
                               verbose=1,
                               n_jobs=2)

    start = time.time()
    grid_search.fit(X_train, y_train)
    print("Run time = ", time.time() - start)
    return grid_search


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

data_record = {"trainning_time_consume(s)": [], "test_time_consume(s)": []}

import argparse
# print(a)
from mpi4py import MPI


def grid_i(X_train, y_train):
    grid_search = GridSearchCV(TabNetRegressor(objective='reg:squarederror', n_jobs=1, random_state=42),
                               param_grid,
                               cv=3,
                               scoring='neg_mean_squared_error',
                               return_train_score=True,
                               verbose=1,
                               n_jobs=2)

    start = time.time()
    grid_search.fit(X_train, y_train)
    print("Run time = ", time.time() - start)
    return grid_search


def model_cv(**kwargs):
    X = {}
    y = {}
    n_epochs = 200
    X['train'], X['val'], y['train'], y['val'] = sklearn.model_selection.train_test_split(
        X_train, y_train, train_size=0.8
    )
    X['test']= X_test
    y['test']=y_test

    # not the best way to preprocess features, but enough for the demonstration
    preprocess = sklearn.preprocessing.StandardScaler().fit(X['train'])
    X = {
        k: torch.tensor(preprocess.fit_transform(v), device=device)
        for k, v in X.items()
    }
    y = {k: torch.tensor(v, device=device) for k, v in y.items()}

    # !!! CRUCIAL for neural networks when solving regression problems !!!
    if task_type == 'regression':
        y_mean = y['train'].mean().item()
        y_std = y['train'].std().item()
        y = {k: (v - y_mean) / y_std for k, v in y.items()}
    else:
        y_std = y_mean = None

    if task_type != 'multiclass':
        y = {k: v.float() for k, v in y.items()}


    model =rtdl.FTTransformer.make_default(
    n_num_features=10,
    cat_cardinalities=None,
    last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
    d_out=6,
)
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
    def evaluate(part):
        model.eval()
        prediction = []
        for batch in zero.iter_batches(X[part], 1024):
            prediction.append(apply_model(batch))
        prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
        target = y[part].cpu().numpy()

        if task_type == 'binclass':
            prediction = np.round(scipy.special.expit(prediction))
            score = sklearn.metrics.accuracy_score(target, prediction)
        elif task_type == 'multiclass':
            prediction = prediction.argmax(1)
            score = sklearn.metrics.accuracy_score(target, prediction)
        else:
            assert task_type == 'regression'
            score = sklearn.metrics.mean_squared_error(target, prediction)
        return score

    batch_size = 256
    train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)

    # Create a progress tracker for early stopping
    # Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html

    progress = zero.ProgressTracker(patience=100)
    start_train = time.time()

    report_frequency = len(X['train']) // batch_size // 5
    start_train = time.time()
    for epoch in range(1, n_epochs + 1):
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

        val_score = evaluate('val')
        test_score = evaluate('test')
        print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
    train_time = time.time() - start_train

    start_pred = time.time()
    test_score = evaluate('test')
    test_time = time.time() - start_pred

    data_record["trainning_time_consume(s)"].append(train_time)
    data_record["test_time_consume(s)"].append(test_time)
    return -test_score


def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]
    print(X_train.shape)
    print(model_save_path + get_related_path(Material_ID).replace(".csv", ".json"))

    rf_bo = BayesianOptimization(
        model_cv,
        {
            "n_d": [32, 128],
            "n_a":[32,512],
            "n_steps":[1,5],
            "gamma":[0.5,2],
            "lambda_sparse":[0.1,1],
            "n_independent":[1.5,4.5],
            "n_shared":[0.5,2.5]

        }
    )

    rf_bo.maximize(init_points=5,n_iter=num_of_iteration)

    #save data
    result_data_root = "." + os.sep + "BO_result_data" + os.sep
    routing_data_root = "." + os.sep + "BO_training_routing" + os.sep
    pd.DataFrame(data_record)
    routing_data_root + get_related_path(Material_ID)
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
    data_record["trainning_time_consume(s)"].clear()
    data_record["test_time_consume(s)"].clear()


def run_Grid_search(num_of_iteration):
    print(num_of_iteration)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start, end = get_range(mix_index)

    product_index = list(itertools.product(data_set_index, list(range(start, end))))
    print(product_index)
    print("total size", len(product_index))
    for index in range(rank, len(data_set_index), size):
        data_index = data_set_index[index]
        print(data_index)

        model_save_path = "." + os.sep + "saved_model" + os.sep + "mini_dataset_mixture" + os.sep + f"mini_data_{data_index}" + os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root + f"mini_data_{data_index}" + os.sep
        saved_root = "." + os.sep + "mini_cleaned_data_mixture_"+device + os.sep + f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(1, 2)
