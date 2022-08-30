import os
import pandas as pd
dataset=[0,1,2,3]

for data_index in dataset:
    data_root="."+os.sep+f"mini_data_{data_index}"+os.sep+"BO_result_data"+os.sep+"mix_2"+os.sep+'mixture.csv'

    data= pd.read_csv(data_root,comment="#",index_col=[0])
    pd.set_option('display.max_columns', None)
    #显示所有行
    pd.set_option('display.max_rows', None)
    for i in range(len(data)):
        correct=eval(data.iloc[i]["params"])
        del correct['n_shared']
        del correct['n_independent']
        del correct['lambda_sparse']
        data.loc[i,"params"]=str(correct)
        print(data.iloc[i])
        print(i)

    data.to_csv(data_root)