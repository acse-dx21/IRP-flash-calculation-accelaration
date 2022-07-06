from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, PRMIX, \
    FlashVL
from torch.utils.data import DataLoader
from thermo import FlashPureVLS

from thermo.interaction_parameters import IPDB
from thermo import SRKMIX, FlashVLN
from torch.utils.data import Dataset

import torch
import numpy as np
import pandas as pd
import sys

sys.path.append("..")
from tool.log import TPZs_log


class flashdata_modified(Dataset):
    """
    dataset of output: phase component of
    """

    def __init__(self, constants, properties, pd_conditions ,Systems="Vapor_Liquid"):
        '''
        :param constants: several meterial
        :param properties: meterial properties
        :param conditions: dictionary:{condition1: condition1_list,condition2: condition2_list}

        :param zs_list: Mole fractions of each component list, required unless there is only
            one component, [-]
        :param Systems: one of ["pure","Vapor_Liquid","Vapor_Multi_Liquid"]
        example:
        >>> constants, properties = ChemicalConstantsPackage.from_IDs(['methane', 'ethane', 'nitrogen'])
        >>> a=flashdata(constants,properties,{"T":[177.5,120.0],"P":[1e5,2e5]},[[.25, 0.7, .05],[.25, 0.7, .05]],"Vapor_Liquid")
        >>> train_loader = DataLoader(a, shuffle=True,batch_size=1,collate_fn=collate_VL)
        >>> a[0]
        EquilibriumState(T=177.5, P=100000.0, zs=[0.25, 0.7, 0.05], betas=[0.9701227587661881, 0.029877241233811858], gas=<CEOSGas, T=177.5 K, P=100000 Pa>, liquids=[CEOSLiquid(eos_class=PRMIX, eos_kwargs={"Pcs": [4599000.0, 4872000.0, 3394387.5], "Tcs": [190.564, 305.32, 126.2], "omegas": [0.008, 0.098, 0.04], "kijs": [[0.0, -0.0059, 0.0289], [-0.0059, 0.0, 0.0533], [0.0289, 0.0533, 0.0]]}, HeatCapacityGases=[HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="TRCIG"), HeatCapacityGas(CASRN="74-84-0", MW=30.06904, similarity_variable=0.26605438683775734, extrapolation="linear", method="TRCIG"), HeatCapacityGas(CASRN="7727-37-9", MW=28.0134, similarity_variable=0.07139440410660612, extrapolation="linear", method="TRCIG")], T=177.5, P=100000.0, zs=[0.009522975422008546, 0.9902824411636479, 0.00019458341434332182])], solids=[])
        '''

        super(flashdata_modified, self).__init__()
        assert Systems in ["pure", "Vapor_Liquid", "Vapor_Liquid_Multi"]


        self.condition=pd.DataFrame(pd_conditions)
        self.len=len(self.condition)

        if len(constants.names) == 1:
            Systems = "pure"

        if Systems == "Vapor_Liquid_Multi":
            self.eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
            self.gas = CEOSGas(SRKMIX, self.eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
            self.liq = CEOSLiquid(SRKMIX, self.eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
            self.flash = FlashVLN(constants, properties, liquids=[self.liq, self.liq], gas=self.gas)
        elif Systems == "Vapor_Liquid":
            self.kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')
            self.eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas,
                               'kijs': self.kijs}
            self.gas = CEOSGas(PRMIX, eos_kwargs=self.eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
            self.liq = CEOSLiquid(PRMIX, eos_kwargs=self.eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
            self.flash = FlashVL(constants, properties, liquid=self.liq, gas=self.gas)
        elif Systems == "pure":
            self.eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
            self.liq = CEOSLiquid(PRMIX, HeatCapacityGases=properties.HeatCapacityGases, eos_kwargs=self.eos_kwargs)
            self.gas = CEOSGas(PRMIX, HeatCapacityGases=properties.HeatCapacityGases, eos_kwargs=self.eos_kwargs)
            self.flash = FlashPureVLS(constants, properties, gas=self.gas, liquids=[self.liq], solids=[])

    def __getitem__(self, idx):
        dic=self.condition.iloc[idx].to_dict()
        dic["zs"] = eval(dic.pop("zs"))
        return self.flash.flash(**dic)

    def __len__(self):
        return self.len

class flashdata(Dataset):
    """
    dataset of output: phase component of
    """

    def __init__(self, constants, properties, conditions, zs_list, Systems="Vapor_Liquid"):
        '''
        :param constants: several meterial
        :param properties: meterial properties
        :param conditions: dictionary:{condition1: condition1_list,condition2: condition2_list}
        :param T_list: Temperature list [K]s
        :param P_list: presure list   [Pa]s
        :param zs_list: Mole fractions of each component list, required unless there is only
            one component, [-]
        :param Systems: one of ["pure","Vapor_Liquid","Vapor_Multi_Liquid"]
        example:
        >>> constants, properties = ChemicalConstantsPackage.from_IDs(['methane', 'ethane', 'nitrogen'])
        >>> a=flashdata(constants,properties,{"T":[177.5,120.0],"P":[1e5,2e5]},[[.25, 0.7, .05],[.25, 0.7, .05]],"Vapor_Liquid")
        >>> train_loader = DataLoader(a, shuffle=True,batch_size=1,collate_fn=collate_VL)
        >>> a[0]
        EquilibriumState(T=177.5, P=100000.0, zs=[0.25, 0.7, 0.05], betas=[0.9701227587661881, 0.029877241233811858], gas=<CEOSGas, T=177.5 K, P=100000 Pa>, liquids=[CEOSLiquid(eos_class=PRMIX, eos_kwargs={"Pcs": [4599000.0, 4872000.0, 3394387.5], "Tcs": [190.564, 305.32, 126.2], "omegas": [0.008, 0.098, 0.04], "kijs": [[0.0, -0.0059, 0.0289], [-0.0059, 0.0, 0.0533], [0.0289, 0.0533, 0.0]]}, HeatCapacityGases=[HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="TRCIG"), HeatCapacityGas(CASRN="74-84-0", MW=30.06904, similarity_variable=0.26605438683775734, extrapolation="linear", method="TRCIG"), HeatCapacityGas(CASRN="7727-37-9", MW=28.0134, similarity_variable=0.07139440410660612, extrapolation="linear", method="TRCIG")], T=177.5, P=100000.0, zs=[0.009522975422008546, 0.9902824411636479, 0.00019458341434332182])], solids=[])
        '''

        super(flashdata, self).__init__()
        assert Systems in ["pure", "Vapor_Liquid", "Vapor_Liquid_Multi"]

        self.T_list = None
        self.P_list = None
        self.H_list = None
        self.S_list = None

        self.condition_len = []
        for key in conditions.keys():
            assert key in ["T", "P", "H", "S"]
            if key == "T":
                self.T_list = conditions["T"]
                self.condition_len.append(len(self.T_list))
            if key == "P":
                self.P_list = conditions["P"]
                self.condition_len.append(len(self.P_list))
            if key == "H":
                self.H_list = conditions["H"]
                self.condition_len.append(len(self.H_list))
            if key == "S":
                self.S_list = conditions["S"]
                self.condition_len.append(len(self.S_list))
        else:
            assert len(self.condition_len) == 2
            assert self.condition_len[0] == self.condition_len[1]
            self.len = self.condition_len[0]

        self.zs_list = zs_list
        self.get_item_function = lambda idx: NotImplemented

        if len(constants.names) == 1:
            Systems = "pure"

        if Systems == "Vapor_Liquid_Multi":
            self.eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
            self.gas = CEOSGas(SRKMIX, self.eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
            self.liq = CEOSLiquid(SRKMIX, self.eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
            self.flash = FlashVLN(constants, properties, liquids=[self.liq, self.liq], gas=self.gas)
        elif Systems == "Vapor_Liquid":
            self.kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')
            self.eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas,
                               'kijs': self.kijs}
            self.gas = CEOSGas(PRMIX, eos_kwargs=self.eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
            self.liq = CEOSLiquid(PRMIX, eos_kwargs=self.eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
            self.flash = FlashVL(constants, properties, liquid=self.liq, gas=self.gas)
        elif Systems == "pure":
            self.eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
            self.liq = CEOSLiquid(PRMIX, HeatCapacityGases=properties.HeatCapacityGases, eos_kwargs=self.eos_kwargs)
            self.gas = CEOSGas(PRMIX, HeatCapacityGases=properties.HeatCapacityGases, eos_kwargs=self.eos_kwargs)
            self.flash = FlashPureVLS(constants, properties, gas=self.gas, liquids=[self.liq], solids=[])

    def __getitem__(self, idx):
        return self.flash.flash(zs=self.zs_list[idx] if self.zs_list is not None else None,
                                T=self.T_list[idx] if self.T_list is not None else None,
                                P=self.P_list[idx] if self.P_list is not None else None,
                                H=self.H_list[idx] if self.H_list is not None else None,
                                S=self.S_list[idx] if self.S_list is not None else None)

    def __len__(self):
        return self.len


class to_numpy:
    def __init__(self):
        self.numpy = []

    def __call__(self, list):
        self.numpy.append(list)

    def release(self):
        return np.array(self.numpy, dtype=np.float32)


def _collect_gas_liq_phase(flash):
    Material_num = len(flash.names)

    if Material_num > 1:
        if not hasattr(flash, "gas") or ((hasattr(flash, "gas") and (flash.gas is None))):
            gas = np.zeros(Material_num)
            temp_output = np.concatenate((gas, np.array(flash.liquid0.zs)), 0)
        if not hasattr(flash, "liquid0") or ((hasattr(flash, "liquid0") and (flash.liquid0 is None))):
            liquid0 = np.zeros(Material_num)
            temp_output = np.concatenate((np.array(flash.gas.zs), liquid0), 0)
        if hasattr(flash, "liquid0") and (flash.liquid0 is not None) and (hasattr(flash, "gas")) and (
                flash.gas is not None):
            temp_output = np.concatenate((np.array(flash.gas.zs), np.array(flash.liquid0.zs)), 0)

        beta = flash.betas
        if len(beta) != 2:
            beta = np.array([1, 0]) if sum(temp_output[:Material_num]) > 0.99 else np.array([0, 1])
        return np.concatenate((temp_output, beta), 0)

    else:
        if hasattr(flash, "liquids"):
            if len(flash.liquids) > 0:
                result = [0, 1, 0, 1]
                return np.array(result)
        if hasattr(flash, "gas"):
            result = [1, 0, 1, 0]
            return np.array(result)

        print("last Mix_1 collecton doesnt not have liquid and gas attribute")
        raise RuntimeError


def collate_VL(batch):
    output = to_numpy()
    input = to_numpy()

    for flash in batch:
        # print(flash)
        input(np.concatenate(
            (np.array(flash.constants.Pcs), np.array(flash.constants.Tcs), np.array(flash.constants.omegas), [flash.T],
             [flash.P], flash.zs), 0))

        output(_collect_gas_liq_phase(flash))

    return torch.tensor(input.release()), torch.tensor(output.release())


def collate_VL_NParray(batch):
    output = to_numpy()
    input = to_numpy()

    for flash in batch:
        # print(flash)

        input(np.concatenate(
            (np.array(flash.constants.Pcs), np.array(flash.constants.Tcs), np.array(flash.constants.omegas), [flash.T],
             [flash.P], flash.zs), 0))

        output(_collect_gas_liq_phase(flash))

    return np.array(input.release()), np.array(output.release())


def genertate_T(size, min, max):
    return list(np.random.rand(size) * (max - min) + min)


def genertate_P(size, min, max):
    return list(np.random.rand(size) * (max - min) + min)


def genertate_Zs_n(size, n):
    result = []
    for i in range(size):
        row_data = np.random.dirichlet(np.ones(n), size=1)
        result.append(list(row_data[0]))
    return result


def generate_good_TPZ(target_size, constants, properties, file_name, comment="No comment"):
    """
    genertate T,P,Zs and record data to good_TPZ.csv
    """
    T_list_good = []
    P_list_good = []
    Zs_list_good = []
    cnt = 0
    size = 200
    material_num = len(constants.names)
    while ((len(T_list_good) < target_size) and (cnt < 1000)):
        cnt += 1
        T_list = genertate_T(size, min=10, max=1000)
        P_list = genertate_P(size, min=1e5, max=10e5)
        Zs_list = genertate_Zs_n(size, material_num)

        test = flashdata(constants, properties, {"T": T_list, "P": P_list}, Zs_list, "Vapor_Liquid")
        for i in range(size):
            try:
                test[i]  # test if good

                # good
                T_list_good.append(T_list[i])
                P_list_good.append(P_list[i])
                Zs_list_good.append(Zs_list[i])

            except:
                # bad
                pass
            continue

    data = pd.DataFrame(dict(zip(["T", "P", "zs"], [T_list_good, P_list_good, Zs_list_good])))
    log = TPZs_log(file_name)
    log(data)
    log.release("#" + comment)

    return T_list_good, P_list_good, Zs_list_good


class Generate_data_from_csv:
    """
    generate data from csv file
    use to_XXX function to get target type of data
    the path should directly related to train and test
    """
    train_test_rate = 0.1

    @staticmethod
    def read_good_TPZ_from_csv(path):
        data = pd.read_csv(path, comment='#')

        T_set = data["T"].tolist()
        P_set = data["P"].tolist()
        Zs_set = data["zs"].tolist()
        for i in range(Zs_set.__len__()):
            Zs_set[i] = eval(Zs_set[i])
        return T_set, P_set, Zs_set

    def __init__(self, path_train, path_test):

        self.T_set_train_all, self.P_set_train_all, self.Zs_set_train_all = Generate_data_from_csv.read_good_TPZ_from_csv(
            path_train)
        self.T_set_test_all, self.P_set_test_all, self.Zs_set_test_all = Generate_data_from_csv.read_good_TPZ_from_csv(
            path_test)


        path_train_ID = eval(open(path_train).readlines()[-1].replace("#", "").replace("\\n", "").replace("/n", "").replace("  ", ","))
        path_test_ID = eval(open(path_test).readlines()[-1].replace("#", "").replace("\\n", "").replace("/n", "").replace("  ", ","))


        # self.T_set_train_all, self.P_set_train_all, self.Zs_set_train_all=np.array(self.T_set_train_all),np.array(self.P_set_train_all),np.array(self.Zs_set_train_all)
        # self.T_set_test_all, self.P_set_test_all, self.Zs_set_test_all=np.array(self.T_set_test_all),np.array(self.P_set_test_all),np.array(self.Zs_set_test_all)
        assert path_test_ID == path_train_ID
        self.material_ID = path_train_ID

        self.system = "pure" if len(self.material_ID) == 1 else "Vapor_Liquid"



    def to_numpy(self):
        # if data_max== None:
        #     data_max=len(self.T_set_train_all)
        T_set_train, P_set_train, Zs_set_train = self.T_set_train_all, self.P_set_train_all, self.Zs_set_train_all
        # test_deep=data_max*Generate_data_from_csv.train_test_rate

        T_set_test, P_set_test, Zs_set_test = self.T_set_test_all, self.P_set_test_all, self.Zs_set_test_all
        constants, properties = ChemicalConstantsPackage.from_IDs(self.material_ID)

        train_set = flashdata(constants, properties, {"T": T_set_train, "P": P_set_train}, Zs_set_train,
                              self.system)
        test_set = flashdata(constants, properties, {"T": T_set_test, "P": P_set_test}, Zs_set_test,
                             self.system)

        train_loader = DataLoader(train_set, shuffle=False,
                                  batch_size=train_set.__len__(), collate_fn=collate_VL_NParray)
        test_loader = DataLoader(test_set, shuffle=False,
                                 batch_size=test_set.__len__(), collate_fn=collate_VL_NParray)

        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))
        return X_train, y_train, X_test, y_test

    def to_dataloader(self, batch_size,collect_fn=collate_VL):
        # if data_max== None:
        #     data_max=len(self.T_set_train_all)
        T_set_train, P_set_train, Zs_set_train = self.T_set_train_all, self.P_set_train_all, self.Zs_set_train_all
        T_set_test, P_set_test, Zs_set_test = self.T_set_test_all, self.P_set_test_all, self.Zs_set_test_all
        constants, properties = ChemicalConstantsPackage.from_IDs(self.material_ID)

        train_set = flashdata(constants, properties, {"T": T_set_train, "P": P_set_train}, Zs_set_train,
                              self.system)
        test_set = flashdata(constants, properties, {"T": T_set_test, "P": P_set_test}, Zs_set_test,
                             self.system)

        train_loader = DataLoader(train_set, shuffle=False,
                                  batch_size=batch_size, collate_fn=collect_fn)
        test_loader = DataLoader(test_set, shuffle=False,
                                 batch_size=batch_size, collate_fn=collect_fn)

        return train_loader, test_loader


import os

class multicsv_data_generater:
    """
    generate all data from single file
    """

    def __init__(self, file_path_root=os.path.dirname(__file__)+os.sep+"cleaned_data"+os.sep,return_type="NParray", transform=None):

        self.file_path = file_path_root
        self.transform = transform
        self.csv_train = {}
        self.csv_test = {}
        self.labels_train = []
        self.labels_test = []

        self.materials = ()

        self.batch_size=512

        self.category = {}
        cnt = -1
        self.return_type=return_type
        # store all the dictionary path of every picture, and the corresponding label of it
        for root, _, fnames in os.walk(file_path_root):
            self.category[root] = cnt + 1
            for fname in fnames:
                csv_names = os.path.join(root, fname)  # find relavent path of every picture
                if "train" in fname:
                    ID = eval(fname.replace("_train.csv", ""))
                    self.csv_train[ID] = csv_names  # path of image
                    self.labels_train.append(ID)

                elif "test" in fname:
                    ID = eval(fname.replace("_test.csv", ""))
                    self.csv_test[ID] = csv_names  # path of image
                    self.labels_test.append(ID)
            cnt += 1

        self.materials = tuple(self.labels_train)

        del self.category[file_path_root]
    def __getitem__(self, idx):
        """

        :param idx:tuple:tuple of material, int element in self.material
        :param Datatype: one of "Dataloader" and "NParray"
        :return:
        """
        if isinstance(idx, tuple):
            target_data = Generate_data_from_csv(self.csv_train[idx], self.csv_test[idx])
            material_ID = idx

        if isinstance(idx, int):
            target_data = Generate_data_from_csv(self.csv_train[self.materials[idx]],
                                                 self.csv_test[self.materials[idx]])
            material_ID = self.materials[idx]

        if self.return_type == "Dataloader":
            train_loader, test_loader = target_data.to_dataloader(self.batch_size)
            return train_loader, test_loader, material_ID
        elif self.return_type == "NParray":
            X_train, y_train, X_test, y_test = target_data.to_numpy()
            return X_train, y_train, X_test, y_test, material_ID
        else:
            print("Datatype in multicsv_data_generater.__getitem__(Datatype) shoulb be one of 'Dataloader' , 'NParray'")
            raise RuntimeError

    def __len__(self):
        return len(self.materials)

    def set_return_type(self,return_type):
        assert return_type in ["Dataloader","NParray"]
        self.return_type=return_type

    def set_batch_size(self,batch_size):
        assert batch_size>0
        self.batch_size=batch_size
    @property
    def get_return_type(self):
        return self.return_type
