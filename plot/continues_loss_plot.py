import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

path1 = "E:\Ic document\IRP-Accelerating-flash-calculation-through-deep-learn\Simple_ANN_experience\My_Mass_Balance_loss_data"
path2 = "E:\Ic document\IRP-Accelerating-flash-calculation-through-deep-learn\Simple_ANN_experience\MSELoss_data"
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
datasize={"single_predict":1,"mini_data_0":2100,"mini_data_1":21199,"mini_data_2":42000,"mini_data_3":63000}
meaningfull_word={"target":"MSELoss","test_time_consume(s)":"test_time_consume(s)"}


data_root=".."+os.sep+"XGB_experience"+os.sep+"mini_cleaned_data_mixture_cuda"+os.sep+"mini_data_0"+os.sep+"continues_pred"+os.sep+"mix_2"+os.sep+"mixture_norm.csv"

data=pd.read_csv(data_root)
data
data.columns=['iteration', 'MSEloss']
sns.lineplot(data=data[:25],y="MSEloss",x="iteration")

plt.show()