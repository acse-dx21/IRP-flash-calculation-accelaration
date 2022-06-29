import numpy as np
import inspect
from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, PRMIX, \
    FlashVL
from torch.utils.data import DataLoader
from thermo import FlashPureVLS
from data.generate_data import genertate_T, genertate_P, genertate_Zs_n,flashdata_modified,collate_VL,Generate_data_from_csv
from thermo.interaction_parameters import IPDB
from thermo import SRKMIX, FlashVLN
from torch.utils.data import Dataset

import torch
import numpy as np
import pandas as pd
import sys

from thermo import FlashPureVLS
class Base:
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any

        init = getattr(cls.__init__, "deprecated_original",cls.__init__)

        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)



        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
    def get_params(self, deep=True):

        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()

                out.update((key + "__" + k, val) for k, val in deep_items)

            out[key] = value
        return out
class t2(FlashPureVLS,Base):
    def __init__(self):
        super().__init__()


def test(key,value,c=2):
    print(key,value,c)





# import argparse
# parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--a', type=str, default = True)
# parser.add_argument('--b', type=int, default=32)
# parser.add_argument('--c', type=int, default=32)
# args = parser.parse_args()
# print(vars(args))
# test(**vars(args))

# print(a.get_params)


import pandas as pd

datas = pd.DataFrame([['a', 1], ['b', 2]], columns=['key', 'value'])
print(datas)
dict1 = dict(zip(datas['key'], datas['value']))
print(dict1)
print(test(**datas.iloc[1].to_dict()))

constants, properties = ChemicalConstantsPackage.from_IDs(['methane', 'ethane', 'nitrogen'])
import time

#
# a=flashdata_modified(constants,properties,{"T":[177.5,120.0],"P":[1e5,2e5],"zs":[[.25, 0.7, .05],[.25, 0.7, .05]]},"Vapor_Liquid")
# train_loader = DataLoader(a, shuffle=True,batch_size=1,collate_fn=collate_VL)
# print(a[1])
import
start=time.time()
import os
# print(os.getcwd())
file=open(".\\data\\mini_cleaned_data\\train_3000\\mix_3\\('Ethane', 'N-Butane', 'N-Hexane')_train.csv")
print(file.readlines()[0])

# datas=Generate_data_from_csv(".\\data\\mini_cleaned_data\\train_3000\\mix_3\\('Ethane', 'N-Butane', 'N-Hexane')_train.csv",".\\data\\mini_cleaned_data\\test_1000\\mix_3\\('Ethane', 'N-Butane', 'N-Hexane')_test.csv")
# datas.to_numpy()
# print(time.time()-start)
