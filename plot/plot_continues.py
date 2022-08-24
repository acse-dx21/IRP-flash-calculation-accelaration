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

datasize = {"single_predict": 1, "mini_data_0": 2100, "mini_data_1": 21000, "mini_data_2": 42000, "mini_data_3": 63000}
meaningfull_word = {"target": "MSELoss", "test_time_consume(s)": "test_time_consume(s)"}


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


import matplotlib.ticker as ticker


class line_bar_plot_from_csv_norm2:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.fig, self.ax = plt.subplots()
        self.colors = ['pink', 'lightgreen', 'lightblue']
        self.comment = ["norm", "noPreprocess"]
        self.meaningful_word = {"norm": "norm", "noPreprocess": "no_standarlization"}
        self.cnt = 0

    def collect_file_path(self, root):
        collected_file = []
        for root, _, files in os.walk(root):
            for file in files:
                collected_file.append(os.path.join(root, file))

        return collected_file

    def _collect_data(self, path, target):
        # print(path,pd.read_csv(path, index_col=0,comment="#")[target].abs().min())
        return pd.read_csv(path, index_col=0, comment="#")[target].abs()

    def _collect_data_index(self, path, target):
        # print(path,pd.read_csv(path, index_col=0,comment="#")[target].abs().min())
        return pd.read_csv(path, comment="#")[target].abs()

    def structed_add_files(self, root, target, norm=None):
        """
        add Dataframe files,with key = mix_i and values= dataframe(comb(materials datas))
        :param root:
        :return:data {c_i:
        """
        comment = " "

        for i in self.comment:
            if i in root:
                comment += "_" + i
        if "cuda" in root or "gpu" in root:
            comment += "_gpu"
        if "cpu" in root or "CPU" in root:
            comment += "_cpu"
        cnt = 0
        # comment += str(self.cnt)
        data_in_this_model = []
        for roots, dir, files in os.walk(root):
            sigle_material_data = []

            for file in files:
                print(roots, file)

                for data in datasize.keys():
                    if data in roots:
                        target_size = datasize[data]
                if norm == "continues":
                    reformed_data = {target: self._collect_data(os.path.join(roots, file), target)}
                    reformed_data = {**reformed_data,
                                     **{"mixture": roots.split(os.sep)[-1], "material": file.replace(".csv", ""),
                                        "model": root.split(os.sep)[1].replace("experience", "") + comment,
                                        "data_size": target_size}}

                elif norm == "min":
                    reformed_data = {target: [self._collect_data(os.path.join(roots, file), target).min()]}
                    reformed_data = {**reformed_data,
                                     **{"mixture": roots.split(os.sep)[-1], "material": file.replace(".csv", ""),
                                        "model": root.split(os.sep)[1].replace("experience", "") + comment,
                                        "data_size": target_size}}
                else:

                    reformed_data = {target: self._collect_data(os.path.join(roots, file), target)}
                    reformed_data = {**reformed_data,
                                     **{"mixture": roots.split(os.sep)[-1], "material": file.replace(".csv", ""),
                                        "model": root.split(os.sep)[1].replace("experience", "") + comment,
                                        "data_size": target_size}}

                sigle_material_data.append(pd.DataFrame(reformed_data))

            try:

                data_in_this_model.append(pd.concat(sigle_material_data, axis=0, ignore_index=True))

            except:
                pass
            cnt += 1
            # print(root)

        print("datalength", len(self.data))
        if (len(data_in_this_model) > 1):
            self.data.append(pd.concat(data_in_this_model, axis=0, ignore_index=True))
        else:
            self.data.append(*data_in_this_model)

        self.cnt += 1
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

    def plot_box(self, x, y):

        multi_model_data = pd.concat(self.data, axis=0, ignore_index=True)

        multi_model_data = multi_model_data.loc[multi_model_data["mixture"] == "mix_2"]
        sns.catplot(x=x, y=y,
                    hue="model",
                    data=multi_model_data, kind="box", height=6, ci=96, margin_titles=True,
                    aspect=1.7)

        plt.ylabel(meaningfull_word[y])
        plt.xlabel(x)

    def plot_line(self, x, y):

        multi_model_data = pd.concat(self.data, axis=0, ignore_index=True)
        print(multi_model_data)
        multi_model_data = multi_model_data.loc[multi_model_data["mixture"] == "mix_2"]
        sns.lineplot(data=multi_model_data, x=x, y=y, hue="model", style="model", markers=True, dashes=False)
        plt.ylabel(meaningfull_word[y])
        plt.xlabel(x)
        plt.yscale("log")
        plt.ylim(0, 0.5)
        sns.set()

    def plot_continues(self, x, y):
        print(self.data[0][:130])
        multi_model_data = pd.concat(self.data, axis=0, ignore_index=False).reset_index()
        multi_model_data = multi_model_data.loc[multi_model_data["index"] < 150]
        print(multi_model_data[multi_model_data["model"] == "tabnet_ gpu"])
        print(multi_model_data)
        multi_model_data = multi_model_data.loc[multi_model_data["mixture"] == "mix_2"]
        sns.lineplot(data=multi_model_data, x=x, y=y, hue="model", style="model", markers=True, dashes=False)
        plt.ylabel(meaningfull_word[y])
        plt.xlabel(x)
        plt.yscale("log")
        plt.ylim(0, 0.5)
        sns.set()

        # plt.xscale("log")

    def plot_bar(self, x, y, hue):

        multi_model_data = pd.concat(self.data, axis=0, ignore_index=True)
        print(multi_model_data)
        multi_model_data = multi_model_data.loc[multi_model_data["mixture"] == "mix_2"]
        sns.barplot(data=multi_model_data, x=x, y=y, hue=hue, )

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

final_root = "continues_pred"

tab_root11 = f"..\\tabnet_experience\\mini_cleaned_data_mixture_cuda_noPreprocess\\mini_data_2\\{final_root}\\"

tab_root12 = f"..\\tabnet_experience\\mini_cleaned_data_mixture_cuda\\mini_data_2\\{final_root}\\"

LGBM_root9 = f"..\\LightGBM_experience\\mini_cleaned_data_mixture_cpu_noPreprocess\\mini_data_2\\{final_root}\\"

LGBM_root2 = f"..\\LightGBM_experience\\mini_cleaned_data_mixture_cpu\\mini_data_2\\{final_root}\\"

XGB_root13 = f"..\\XGB_experience\\mini_cleaned_data_mixture_cuda_noPreprocess\\mini_data_2\\{final_root}\\"

ANN_root1 = f"..\\Simple_ANN_experience\\mini_cleaned_data_mixture_cuda\\mini_data_2\\{final_root}\\"

FT_root1 = f"..\\FT-transformer\\mini_cleaned_data_mixture_cuda\\mini_data_2\\{final_root}\\"

roots = []
roots.append(tab_root11)

roots.append(tab_root12)

roots.append(LGBM_root9)

# roots.append(LGBM_root2)

roots.append(XGB_root13)

roots.append(ANN_root1)

roots.append(FT_root1)

print(a.data)


# if final_root=="BO_training_routing":
#     roots.append(tab_root15)
#     roots.append(tab_root5)
#     # roots.append(XGB_root17)
#     roots.append(LGBM_root13)
#     roots.append(XGB_root9)
#     roots.append(ANN_root17)
#     roots.append(RF_root17)

def multi_plot(roots, target):
    for root in roots:
        print(root)
        a.structed_add_files(root, target, "continues")

    ax = a.plot_continues("index", target)
    # ax = a.plot_bar("model",target,"data_size")


multi_plot(roots, "target")
# multi_plot(roots,"target")


plt.title("continues_predition")
plt.xlabel("epoch")
plt.show()
