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

data_set_index = [1,2,3]
mix_index = "all"
device = "cuda"
data_root = "." + os.sep + "mini_cleaned_data" + os.sep
save_model = False
save_data = False
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
    n_block=kwargs["n_blocks"]

    ffn_d_hidden = kwargs["ffn_d_hidden"]
    attention_dropout = kwargs["attention_dropout"]  # 0.2,
    ffn_dropout = kwargs["ffn_dropout"]  # 0.1,
    residual_dropout = kwargs["residual_dropout"]

    model = rtdl.FTTransformer.make_baseline(
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

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2, weight_decay=1e-5)
    loss_fn = F.mse_loss

    # def apply_model(x_num, x_cat=None):
    #        return model(x_num, x_cat)

    @torch.no_grad()
    def evaluate(X, y):

        model.eval()
        prediction = []
        for batch in zero.iter_batches(X, 1024):
            # batch.to(device)
            prediction.append(model(batch, None))
        prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
        target = y.cpu().numpy()

        score = np.sqrt(sklearn.metrics.mean_squared_error(target, prediction))

        return score

    # Create a progress tracker for early stopping
    # Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
    batch_size = 256
    train_loader = zero.data.IndexLoader(len(X_train), batch_size)

    n_epochs = 1000
    checkpoint_path = model_save_path + get_related_path(Material_ID).replace(".csv", ".pt")
    # report_frequency = len(X_train_norm) // batch_size // 3
    progress = zero.ProgressTracker(patience=100)
    for epoch in range(1, n_epochs + 1):
        start = time.time()
        model.train()
        for iteration, batch_idx in enumerate(train_loader):
            x_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            loss = loss_fn(model(x_batch, None).squeeze(1), y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if iteration % report_frequency == 0:
            #    print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

        val_score = evaluate(X_val, y_val)
        test_score = evaluate(X_test, y_test)
        if epoch % 10 == 0:
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

        if progress.fail or epoch == n_epochs:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            val_score = evaluate(X_val, y_val)
            test_score = evaluate(X_test, y_test)
            print('Checkpoint recovered:')
            print(
                f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f} | {time.time() - start:.1f}s',
                end='')
            break
    return -test_score


from sklearn.model_selection import train_test_split


def run_bayes_optimize(num_of_iteration=10, data_index=2):
    BO_root = "." + os.sep + "BO_result_data" + os.sep
    global X_train, y_train, X_val, y_val, X_test, y_test, Material_ID
    X_train, y_train, X_test, y_test, Material_ID = relate_data[data_index]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=4)

    # convert to tensor
    X_train = torch.tensor(X_train.astype('float32')).to(device)
    X_test = torch.tensor(X_test.astype('float32')).to(device)
    X_val = torch.tensor(X_val.astype('float32')).to(device)
    y_train = torch.tensor(y_train.astype('float32')).to(device)
    y_test = torch.tensor(y_test.astype('float32')).to(device)
    y_val = torch.tensor(y_val.astype('float32')).to(device)
    print(X_train.shape)
    print(model_save_path + get_related_path(Material_ID).replace(".csv", ".json"))

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

    rf_bo.maximize(init_points=5, n_iter=num_of_iteration)

    # save data
    result_data_root = "." + os.sep + "BO_result_data" + os.sep
    routing_data_root = "." + os.sep + "BO_training_routing" + os.sep
    pd.DataFrame(data_record)
    routing_data_root + get_related_path(Material_ID)
    if save_data:
        if os.path.exists(saved_root + result_data_root + get_related_path(Material_ID)):
            pd.concat([pd.DataFrame(rf_bo.res),
                       pd.read_csv(saved_root + result_data_root + get_related_path(Material_ID), index_col=0,
                                   comment="#")], ignore_index=True).to_csv(
                saved_root + result_data_root + get_related_path(Material_ID))
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

        model_save_path = "." + os.sep + "saved_model" + os.sep + "mini_dataset_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        mini_data_path = ".." + os.sep + "data" + os.sep + data_root + f"mini_data_{data_index}" + os.sep
        saved_root = "." + os.sep + "mini_cleaned_data_mixture_" + device + os.sep + f"mini_data_{data_index}" + os.sep
        All_ID = ['Methane', 'Ethane', 'Propane', 'N-Butane', 'N-Pentane', 'N-Hexane', 'Heptane']
        relate_data = generate_data.mixture_generater(mini_data_path)
        # collector=generate_data.collector()
        # collector.set_collect_method("VF")
        # relate_data.set_collector(collector)
        relate_data.set_batch_size(128)
        relate_data.set_collector("VF")
        run_bayes_optimize(1, 2)
