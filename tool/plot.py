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


def cross_plot(path1, path2):
    files1 = os.listdir(path1)
    files2 = os.listdir(path2)

    label1 = "My_Mass_Balance_loss"
    label2 = "MSELoss"

    length = len(files1)
    plt.figure(dpi=800)
    plt.figure(figsize=(30, 24))
    for i in range(len(files1)):
        data_i = pd.read_csv(path1 + os.sep + files1[i], comment="#")
        data_j = pd.read_csv(path2 + os.sep + files2[i], comment="#")

        plt.subplot(3, int(length / 3), i + 1)
        plt.plot(data_i["epoch"].tolist(), data_i["loss"].tolist(), label=label1)
        plt.plot(data_j["epoch"].tolist(), data_j["loss"].tolist(), label=label2)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title(files1[i])

    plt.show()



class line_bar_plot_from_csv_norm2:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.fig, self.ax = plt.subplots()
        self.colors = ['pink', 'lightgreen', 'lightblue']
        self.comment=["norm"]
        self.cnt=0

    def collect_file_path(self, root):
        collected_file = []
        for root, _, files in os.walk(root):
            for file in files:
                collected_file.append(os.path.join(root, file))

        return collected_file

    def _collect_data(self, path, target):
        print(path,pd.read_csv(path, index_col=0,comment="#"))
        return pd.read_csv(path, index_col=0,comment="#")[target].abs()

    def structed_add_files(self, root, target, norm=None):
        """
        add Dataframe files,with key = mix_i and values= dataframe(comb(materials datas))
        :param root:
        :return:data {c_i:
        """
        comment=" "

        for i in self.comment:
            if i in root:
                comment+="postprocessed_gpu"
        if "cuda" in root or "gpu" in root:
                comment+="gpu"
        if "cpu" in root or "CPU" in root:
                comment+="cpu"
        cnt = 0
        # comment += str(self.cnt)
        data_in_this_model=[]
        for roots, dir, files in os.walk(root):
            sigle_material_data = []

            for file in files:
                print(roots, file)

                for data in datasize.keys():
                    if data in roots:
                        target_size=datasize[data]
                if norm is not None:
                    reformed_data = {target: self._collect_data(os.path.join(roots, file), target) * 1000
                                               / self._collect_data(os.path.join(roots, file), norm)}

                    sigle_material_data.append(pd.DataFrame(reformed_data))
                else:
                    reformed_data={target:self._collect_data(os.path.join(roots, file), target)}
                reformed_data={**reformed_data,
                               **{"mixture":roots.split(os.sep)[-1],"material":file.replace(".csv",""),
                                   "model":root.split(os.sep)[1].replace("experience","")+comment,
                                   "data_size":target_size}}


                sigle_material_data.append(pd.DataFrame(reformed_data))

            try:


                data_in_this_model.append(pd.concat(sigle_material_data, axis=0, ignore_index=True))

            except:
                pass
            cnt += 1
            # print(root)


        if(len(data_in_this_model)>1):
            self.data.append(pd.concat(data_in_this_model,axis=0,ignore_index=True))
        else:
            self.data.append(*data_in_this_model)

        self.cnt+=1
        return self.data

    def plot(self):

        data_rearrange = []
        position = []
        fig, ax = plt.subplots()  # 子图
        # print(self.data[0]["mix_"+str(1)])
        for i in range(7):
            for j in range(len(self.data)):
                try:
                    data_rearrange.append(self.data[j]["C_" + str(i)].dropna(axis=0, how="any").to_numpy())
                    position.append(j * 0.1 + i)
                except:
                    pass
        print(position)
        print(len(data_rearrange[0]))
        ax.boxplot(data_rearrange, positions=position, patch_artist=True, label="")

    def plot_box(self,x,y):

        multi_model_data=pd.concat(self.data,axis=0,ignore_index=True)

        multi_model_data = multi_model_data.loc[multi_model_data["mixture"] == "mix_2"]
        sns.catplot(x=x, y=y,
                    hue="model",
                    data=multi_model_data, kind="box",height=6,ci=96,margin_titles=True,
                     aspect=1.7)

        plt.ylabel(meaningfull_word[y])
        plt.xlabel(x)


    def plot_line(self,x,y):

        multi_model_data = pd.concat(self.data, axis=0, ignore_index=True)
        print(multi_model_data)
        multi_model_data = multi_model_data.loc[multi_model_data["mixture"] == "mix_2"]
        sns.lineplot(data=multi_model_data,x=x, y=y,hue="model",style="model",markers=True, dashes=False)
        plt.ylabel(meaningfull_word[y])
        plt.xlabel(x)
        plt.yscale("log")
        sns.set()
        # plt.xscale("log")

    def plot_bar(self,x,y,hue):

        multi_model_data = pd.concat(self.data, axis=0, ignore_index=True)
        print(multi_model_data)
        multi_model_data = multi_model_data.loc[multi_model_data["mixture"] == "mix_2"]
        sns.barplot(data=multi_model_data,x=x, y=y,hue=hue,)

        plt.ylabel(meaningfull_word[y])
        plt.xlabel(x)
        # plt.yscale("log")
        plt.xticks(rotation=-10)
        # plt.xscale("log")

    def plot_files(self, root):
        files = self.collect_file(root)
        cnt = 0
        collect = {}
        for file in files:
            data = self._collect_data(file)
            cnt += 1
            collect[cnt] = data
        return pd.DataFrame(collect)


a = line_bar_plot_from_csv_norm2("test")
# file="..\\XGB_experience\\BO_result_data\\mix_3\\('Ethane', 'N-Butane', 'N-Pentane').csv"
# root1="..\\Simple_ANN_experience\\BO_result_data\\"
# root2="..\\XGB_experience\\BO_result_data\\"
# root3="..\\LightGBM_experience\\BO_result_data\\"
data_num = 1

# root1 = f"..\\1dcnn_experience\\mini_cleaned_data_mixture\\mini_data_{data_num}\\BO_training_routing\\"

final_root="BO_result_data"

tab_root11=f"..\\tabnet_experience\\mini_cleaned_data_mixture_cpu\\mini_data_0\\{final_root}\\"
tab_root12=f"..\\tabnet_experience\\mini_cleaned_data_mixture_cpu\\mini_data_1\\{final_root}\\"
tab_root13=f"..\\tabnet_experience\\mini_cleaned_data_mixture_cpu\\mini_data_2\\{final_root}\\"
tab_root14=f"..\\tabnet_experience\\mini_cleaned_data_mixture_cpu\\mini_data_3\\{final_root}\\"
tab_root15=f"..\\tabnet_experience\\mini_cleaned_data_mixture_cpu\\single_predict\\{final_root}\\"

tab_root1=f"..\\tabnet_experience\\mini_cleaned_data_mixture_cuda\\mini_data_0\\{final_root}\\"
tab_root2=f"..\\tabnet_experience\\mini_cleaned_data_mixture_cuda\\mini_data_1\\{final_root}\\"
tab_root3=f"..\\tabnet_experience\\mini_cleaned_data_mixture_cuda\\mini_data_2\\{final_root}\\"
tab_root4=f"..\\tabnet_experience\\mini_cleaned_data_mixture_cuda\\mini_data_3\\{final_root}\\"
tab_root5=f"..\\tabnet_experience\\mini_cleaned_data_mixture_cuda\\single_predict\\{final_root}\\"

# root2 = f"..\\tabnet_experience\\mini_cleaned_data_mixture_cuda\\mini_data_0\\BO_result_data\\"
XGB_root5=f"..\\XGB_experience\\mini_cleaned_data_mixture_cuda\\mini_data_0\\{final_root}\\"
XGB_root6=f"..\\XGB_experience\\mini_cleaned_data_mixture_cuda\\mini_data_1\\{final_root}\\"
XGB_root7=f"..\\XGB_experience\\mini_cleaned_data_mixture_cuda\\mini_data_2\\{final_root}\\"
XGB_root8=f"..\\XGB_experience\\mini_cleaned_data_mixture_cuda\\mini_data_3\\{final_root}\\"
XGB_root9=f"..\\XGB_experience\\mini_cleaned_data_mixture_cuda\\single_predict\\{final_root}\\"
# root4=f"..\\Simple_ANN_experience\\mini_cleaned_data\\mini_data_{data_num}\\BO_result_data\\"

LGBM_root9=f"..\\LightGBM_experience\\mini_cleaned_data_mixture_cpu\\mini_data_0\\{final_root}\\"
LGBM_root10=f"..\\LightGBM_experience\\mini_cleaned_data_mixture_cpu\\mini_data_1\\{final_root}\\"
LGBM_root11=f"..\\LightGBM_experience\\mini_cleaned_data_mixture_cpu\\mini_data_2\\{final_root}\\"
LGBM_root12=f"..\\LightGBM_experience\\mini_cleaned_data_mixture_cpu\\mini_data_3\\{final_root}\\"
LGBM_root13=f"..\\LightGBM_experience\\mini_cleaned_data_mixture_cpu\\single_predict\\{final_root}\\"

XGB_root13=f"..\\XGB_experience\\mini_cleaned_data_mixture_cpu\\mini_data_0\\{final_root}\\"
XGB_root14=f"..\\XGB_experience\\mini_cleaned_data_mixture_cpu\\mini_data_1\\{final_root}\\"
XGB_root15=f"..\\XGB_experience\\mini_cleaned_data_mixture_cpu\\mini_data_2\\{final_root}\\"
XGB_root16=f"..\\XGB_experience\\mini_cleaned_data_mixture_cpu\\mini_data_3\\{final_root}\\"
XGB_root17=f"..\\XGB_experience\\mini_cleaned_data_mixture_cpu\\single_predict\\{final_root}\\"

XGB_root20=f"..\\XGB_experience\\mini_cleaned_data_mixture_norm\\mini_data_0\\{final_root}\\"
XGB_root21=f"..\\XGB_experience\\mini_cleaned_data_mixture_norm\\mini_data_1\\{final_root}\\"
XGB_root22=f"..\\XGB_experience\\mini_cleaned_data_mixture_norm\\mini_data_2\\{final_root}\\"
XGB_root23=f"..\\XGB_experience\\mini_cleaned_data_mixture_norm\\mini_data_3\\{final_root}\\"



ANN_root13=f"..\\Simple_ANN_experience\\mini_cleaned_data_mixture_cuda\\mini_data_0\\{final_root}\\"
ANN_root14=f"..\\Simple_ANN_experience\\mini_cleaned_data_mixture_cuda\\mini_data_1\\{final_root}\\"
ANN_root15=f"..\\Simple_ANN_experience\\mini_cleaned_data_mixture_cuda\\mini_data_2\\{final_root}\\"
ANN_root16=f"..\\Simple_ANN_experience\\mini_cleaned_data_mixture_cuda\\mini_data_3\\{final_root}\\"
ANN_root17=f"..\\Simple_ANN_experience\\mini_cleaned_data_mixture_cuda\\single_predict\\{final_root}\\"

ANN_root3=f"..\\Simple_ANN_experience\\mini_cleaned_data_mixture_cpu\\mini_data_0\\{final_root}\\"
ANN_root4=f"..\\Simple_ANN_experience\\mini_cleaned_data_mixture_cpu\\mini_data_1\\{final_root}\\"
ANN_root5=f"..\\Simple_ANN_experience\\mini_cleaned_data_mixture_cpu\\mini_data_2\\{final_root}\\"
ANN_root6=f"..\\Simple_ANN_experience\\mini_cleaned_data_mixture_cpu\\mini_data_3\\{final_root}\\"
# ANN_root7=f"..\\Simple_ANN_experience\\mini_cleaned_data_mixture_cpu\\single_predict\\{final_root}\\"


RF_root13=f"..\\RF_experience\\mini_cleaned_data_mixture_cpu\\mini_data_0\\{final_root}\\"
RF_root14=f"..\\RF_experience\\mini_cleaned_data_mixture_cpu\\mini_data_1\\{final_root}\\"
RF_root15=f"..\\RF_experience\\mini_cleaned_data_mixture_cpu\\mini_data_2\\{final_root}\\"
RF_root16=f"..\\RF_experience\\mini_cleaned_data_mixture_cpu\\mini_data_3\\{final_root}\\"
RF_root17=f"..\\RF_experience\\mini_cleaned_data_mixture_cpu\\single_predict\\{final_root}\\"

FT_root1=f"..\\FT-transformer\\mini_cleaned_data_mixture_cuda\\mini_data_0\\{final_root}\\"
FT_root2=f"..\\FT-transformer\\mini_cleaned_data_mixture_cuda\\mini_data_1\\{final_root}\\"
FT_root3=f"..\\FT-transformer\\mini_cleaned_data_mixture_cuda\\mini_data_2\\{final_root}\\"
FT_root4=f"..\\FT-transformer\\mini_cleaned_data_mixture_cuda\\mini_data_3\\{final_root}\\"

roots=[]
if final_root=="BO_training_routing":
    roots.append(tab_root5)
roots.append(tab_root11)
roots.append(tab_root12)
roots.append(tab_root13)
roots.append(tab_root14)
# tabnet gpu
if final_root=="BO_training_routing":
    roots.append(tab_root5)
roots.append(tab_root1)
roots.append(tab_root2)
roots.append(tab_root3)
roots.append(tab_root4)


#LGBM cpu
if final_root=="BO_training_routing":
    roots.append(LGBM_root13)
roots.append(LGBM_root9)
roots.append(LGBM_root10)
roots.append(LGBM_root11)
roots.append(LGBM_root12)

# xgb gpu
if final_root=="BO_training_routing":
    roots.append(XGB_root9)
roots.append(XGB_root5)
roots.append(XGB_root6)
roots.append(XGB_root7)
roots.append(XGB_root8)

#xgb cpu
# if final_root=="BO_training_routing":
#     roots.append(XGB_root17)
# roots.append(XGB_root13)
# roots.append(XGB_root14)
# roots.append(XGB_root15)
# roots.append(XGB_root16)

# xgb norm
roots.append(XGB_root20)
roots.append(XGB_root21)
roots.append(XGB_root22)
roots.append(XGB_root23)

# #ANN gpu
# if final_root=="BO_training_routing":
#     roots.append(ANN_root17)
# roots.append(ANN_root13)
# roots.append(ANN_root14)
# roots.append(ANN_root15)
# roots.append(ANN_root16)

#ANN cpu
# if final_root=="BO_training_routing":
#     roots.append(ANN_root17)
roots.append(FT_root1)
roots.append(FT_root2)
roots.append(FT_root3)
roots.append(FT_root4)

#RF cpu
# if final_root=="BO_training_routing":
#     roots.append(RF_root17)
# roots.append(RF_root13)
# roots.append(RF_root14)
# roots.append(RF_root15)
# roots.append(RF_root16)




print(a.data)
# if final_root=="BO_training_routing":
#     roots.append(tab_root15)
#     roots.append(tab_root5)
#     # roots.append(XGB_root17)
#     roots.append(LGBM_root13)
#     roots.append(XGB_root9)
#     roots.append(ANN_root17)
#     roots.append(RF_root17)

def multi_plot(roots,target):
    for root in roots:
        print(root)
        a.structed_add_files(root, target)

    ax = a.plot_line("data_size", target)
    # ax = a.plot_bar("model",target,"data_size")

multi_plot(roots,"target")
# multi_plot(roots,"target")





# plt.ylabel(target)
plt.show()
