import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import os
class log:
    def __init__(self,name,columns):
        self.name=name
        self.csv=pd.DataFrame(columns=columns)
        self.cnt = 0
    def __call__(self, pds):
        self.csv=self.csv.append(pds,ignore_index=True)
        self.cnt+=1

    def release(self,comment="No comment"):
        self.csv.to_csv(self.name+".csv",mode="a")
        self.add(comment)

    def add(self,content):
        f=open(self.name+".csv","a+")
        f.write("# "+str(content).replace("#","").replace("\\n"," ").replace("/n","")+os.sep+"n")


class epoch_log(log):
    def __init__(self,name):
        super().__init__(name,columns=["epoch","loss"])


class TPZs_log(log):
    def __init__(self,name):
        super().__init__(name,columns=["T","P","zs"])

class data_num_log(log):
    def __init__(self,name):
        super().__init__(name,columns=["Num_data","score"])
