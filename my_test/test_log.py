import pytest
import sys
sys.path.append("..")
import numpy as np
from tool import log
import pandas as pd

def test_log():
    log_t=log.log("test_log",["test"])
    pds=pd.DataFrame(columns=["test"])
    for i in range(10):
        pds.loc[i]=i
    log_t(pds)
    log_t.release()

    #test
    read_back=pd.read_csv("test_log.csv")["test"]
    for i in range(10):
        assert read_back.loc[i]==i




