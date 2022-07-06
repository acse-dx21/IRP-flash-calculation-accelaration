import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
path1="E:\Ic document\IRP-Accelerating-flash-calculation-through-deep-learn\Simple_ANN_experience\My_Mass_Balance_loss_data"
path2="E:\Ic document\IRP-Accelerating-flash-calculation-through-deep-learn\Simple_ANN_experience\MSELoss_data"
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


class line_bar_plot_from_csv:
    def __init__(self,name):
        self.name=name
        self.data={}
        self.fig, self.ax = plt.subplots()
        self.colors=['pink', 'lightblue', 'lightgreen']
    def collect_file_path(self,root):
        collected_file=[]
        for root, _ , files in os.walk(root):
            for file in files:
                collected_file.append(os.path.join(root,file))

        return collected_file

    def _collect_data(self,path,target):
        return pd.read_csv(path,index_col=0)[target]


    def structed_add_files(self,root,target):
        """
        add Dataframe files,with key = mix_i and values= dataframe(comb(materials datas))
        :param root:
        :return:
        """
        cnt=0
        collected_data={}

        for roots, dir,files in os.walk(root):
            sigle_material_data = []

            for file in files:
                print(file)
                corrct=pd.read_csv(os.path.join(roots, file),index_col=0)
                if "train_time" in corrct.columns:
                    print(corrct)
                    corrct.columns=['trainning_time_consume(s)','test_time_consume(s)']
                    print(corrct)
                    corrct.to_csv(os.path.join(roots, file))
                self._collect_data(os.path.join(roots, file), target)
                sigle_material_data.append(self._collect_data(os.path.join(roots,file),target))
                # print(len(sigle_material_data))
            try:
                df=pd.concat(sigle_material_data,axis=0,ignore_index=True)
                collected_data["C_"+str(cnt)]=df

            except:
                pass
            cnt+=1
            # print(root)
        target=root.split("\\")[1]

        self.data[target]=pd.DataFrame(collected_data)
        return self.data
    def plot(self):
        data_rearrange=[]
        position=[]
        fig, ax = plt.subplots()  # 子图
        # print(self.data[0]["mix_"+str(1)])
        for i in range(7):
            for j in range(len(self.data)):
                try:
                    data_rearrange.append(self.data[j]["C_"+str(i)].dropna(axis=0,how="any").to_numpy()*-1)
                    position.append(j*0.1+i)
                except:
                    pass
        print(position)
        print(len(data_rearrange[0]))
        ax.boxplot(data_rearrange,positions=position,patch_artist=True,label="")

    def plot_box(self):
        cnt=0
        for key in self.data.keys():

            data=[]
            for col in self.data[key].columns:

                data.append(self.data[key][col].dropna(axis=0,how="any").to_numpy()*-1)
            boxprops = {'facecolor': self.colors[cnt]}

            self.ax.boxplot(data,patch_artist=True,boxprops=boxprops)
            cnt += 1

        self.ax.legend(self.data.keys())
        return self.ax

    def plot_line(self,add=False):
        if(add):
            self.ax=self.ax.twinx()

        cnt=0;

        for key in self.data.keys():
            data=[]
            cnt_col=0
            X = []
            for col in self.data[key].columns:
                cnt_col+=1
                data.append(self.data[key][col].dropna(axis=0,how="any").mean())
                X.append(cnt_col)


            print(cnt)
            self.ax.plot(X,data,color=self.colors[cnt])
            cnt += 1


        self.ax.legend(self.data.keys())
        return self.ax



    def plot_files(self,root):
        files=self.collect_file(root)
        cnt=0
        collect={}
        for file in files:

            data=self._collect_data(file)
            cnt+=1
            collect[cnt]=data
        return pd.DataFrame(collect)









a= line_bar_plot_from_csv("test")
# file="..\\XGB_experience\\BO_result_data\\mix_3\\('Ethane', 'N-Butane', 'N-Pentane').csv"
# root1="..\\Simple_ANN_experience\\BO_result_data\\"
# root2="..\\XGB_experience\\BO_result_data\\"
# root3="..\\LightGBM_experience\\BO_result_data\\"

root1="..\\Simple_ANN_experience\\BO_result_data\\"
# root2="..\\XGB_experience\\BO_training_routing\\"
# root3="..\\LightGBM_experience\\BO_training_routing\\"
root4="..\\PINN_experience\\BO_result_data\\"
target="target"
a.structed_add_files(root1,target)
print(a.data)

a.structed_add_files(root4,target)
# a.structed_add_files(root3,target)
ax=a.plot_box()
plt.ylabel(target)
plt.show()