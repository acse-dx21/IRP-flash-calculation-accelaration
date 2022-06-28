import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
path1="E:\Ic document\IRP-Accelerating-flash-calculation-through-deep-learn\Simple_ANN_experience\My_Mass_Balance_loss_data"
path2="E:\Ic document\IRP-Accelerating-flash-calculation-through-deep-learn\Simple_ANN_experience\MSELoss_data"
import matplotlib.pyplot as plt
import pandas as pd
def cross_plot(path1,path2):
    files1 = os.listdir(path1)
    files2 = os.listdir(path2)

    label1="My_Mass_Balance_loss"
    label2= "MSELoss"

    length=len(files1)
    plt.figure(dpi=800)
    plt.figure(figsize=(30,24))
    for i in range(len(files1)):
        data_i=pd.read_csv(path1+os.sep+files1[i],comment="#")
        data_j = pd.read_csv(path2 + os.sep + files2[i], comment="#")

        plt.subplot(3,int(length/3),i+1)
        plt.plot(data_i["epoch"].tolist(),data_i["loss"].tolist(),label=label1)
        plt.plot(data_j["epoch"].tolist(), data_j["loss"].tolist(),label=label2)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title(files1[i])

    plt.show()


cross_plot(path1,path2)