import csv
import pandas as pd
roots=".\\LightGBM_experience\\BO_training_routing\\"
import os
for root, _, files in os.walk(roots):
    for file in files:
        csv_names = os.path.join(root, file)

        print(csv_names)
        path=csv_names
        data=pd.read_csv(path,index_col=0,comment="#")[-10:].reset_index(drop=True)
        data.to_csv(path)
        # try:
        #     data["zs"]=data.pop("Zs")
        #     data.to_csv(path)
        # except:
        #     pass
        # material=eval(file.replace("_train.csv","").replace("_test.csv",""))
        # print(material)
        # file=open(csv_names,"a+")
        # file.write("# "+str(material))
