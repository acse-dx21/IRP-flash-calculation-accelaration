import csv
import numpy as np
import pandas as pd
import random
import os
from mpi4py import MPI
if __name__ == "__main__":
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        datasize=0
        roots="."+os.sep+"data"+os.sep+"cleaned_data"
        rooti="."+os.sep+"data"+os.sep+"mini_cleaned_data"+os.sep+"mini_data_"+str(datasize)
        print(roots)

        for root, dis, files in os.walk(roots):
            for file in files:
                print(file)
                material=eval(file.replace("_test.csv","").replace("_train.csv",""))

                csv_names = os.path.join(root, file)
                indextrain = random.sample(range(50000), 300)
                indextest = random.sample(range(20000), 100)
                # print(csv_names)
                path=csv_names
                data=pd.read_csv(path,index_col=0,comment="#")

                # print(type(data))
                if(len(data)>40000):

                    a=data.iloc[indextrain]
                    a.to_csv(os.path.join(root, file).replace(roots,rooti),mode="w")
                    file=open(os.path.join(root, file).replace(roots,rooti),mode="a")
                    file.write("# "+str(material))
                else:
                    a = data.iloc[indextest]
                    a.to_csv(os.path.join(root, file).replace(roots, rooti),mode="w")
                    file = open(os.path.join(root, file).replace(roots, rooti), mode="a")
                    file.write("# " + str(material))
