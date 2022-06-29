import csv
import pandas as pd
roots=".\\data\\mini_cleaned_data\\"
import os
for root, _, files in os.walk(roots):
    for file in files:
        csv_names = os.path.join(root, file)

        print(csv_names)
        path=csv_names
        data=pd.read_csv(path,index_col=0,comment="#")
        try:
            data["zs"]=data.pop("Zs")
            data.to_csv(path)
        except:
            pass
        material=eval(file.replace("_train.csv","").replace("_test.csv",""))
        print(material)
        file=open(csv_names,"a+")
        file.write("# "+str(material))
