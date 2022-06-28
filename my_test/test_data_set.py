import pytest
import sys

sys.path.append("..")
import numpy as np
from data import generate_data
from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, SRKMIX, FlashVLN, PropertyCorrelationsPackage, \
    HeatCapacityGas, FlashVL, PRMIX, FlashPureVLS
from thermo.interaction_parameters import IPDB
import torch
from torch.utils.data import DataLoader


def test_multi_flash():
    constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Pcs=[4414000.0, 22048320.0, 6137000.0],
                                         omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844],
                                         CASs=['71-36-3', '7732-18-5', '64-17-5'])
    properties = PropertyCorrelationsPackage(constants=constants,
                                             HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0,
                                                                                          [-3.787200194613107e-20,
                                                                                           1.7692887427654656e-16,
                                                                                           -3.445247207129205e-13,
                                                                                           3.612771874320634e-10,
                                                                                           -2.1953250181084466e-07,
                                                                                           7.707135849197655e-05,
                                                                                           -0.014658388538054169,
                                                                                           1.5642629364740657,
                                                                                           -7.614560475001724])),
                                                                HeatCapacityGas(poly_fit=(50.0, 1000.0,
                                                                                          [5.543665000518528e-22,
                                                                                           -2.403756749600872e-18,
                                                                                           4.2166477594350336e-15,
                                                                                           -3.7965208514613565e-12,
                                                                                           1.823547122838406e-09,
                                                                                           -4.3747690853614695e-07,
                                                                                           5.437938301211039e-05,
                                                                                           -0.003220061088723078,
                                                                                           33.32731489750759])),
                                                                HeatCapacityGas(poly_fit=(50.0, 1000.0,
                                                                                          [-1.162767978165682e-20,
                                                                                           5.4975285700787494e-17,
                                                                                           -1.0861242757337942e-13,
                                                                                           1.1582703354362728e-10,
                                                                                           -7.160627710867427e-08,
                                                                                           2.5392014654765875e-05,
                                                                                           -0.004732593693568646,
                                                                                           0.5072291035198603,
                                                                                           20.037826650765965])), ], )
    zs = [[.25, 0.7, .05]]
    # test Temperater,Presure
    conditions0 = {"T": [361], "P": [1e5]}
    flash0 = generate_data.flashdata(constants, properties, conditions0, zs,
                                     "Vapor_Liquid_Multi")[0]
    assert np.allclose(flash0.betas, [0.027939322463004766, 0.6139152961492645, 0.35814538138773067])

    # test H,S
    conditions1 = {"P": [10000.0], "S": [-100]}
    flash1 = generate_data.flashdata(constants, properties, conditions1, zs,
                                     "Vapor_Liquid_Multi")[0]

    assert np.allclose(flash1.betas, [0.07647278963836679, 0.9235272103616332])

    conditions2 = {"P": [10000.0], "H": [-10000]}
    flash2 = generate_data.flashdata(constants, properties, conditions2, zs,
                                     "Vapor_Liquid_Multi")[0]
    assert np.allclose(flash2.betas, [0.7641291069979484, 0.213955738205042, 0.02191515479700956])


def test_VL_flash():
    constants, properties = ChemicalConstantsPackage.from_IDs(['methane', 'ethane', 'nitrogen'])
    kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')

    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
    gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    zs = [0.965, 0.018, 0.017]

    conditions0 = {"T": [110.0], "P": [1e5]}
    flash0 = generate_data.flashdata(constants, properties, conditions0, [zs],
                                     "Vapor_Liquid")[0]

    assert np.allclose(flash0.betas, [0.10365811277428856, 0.8963418872257114])

    conditions1 = {"P": [10000.0], "S": [-100]}
    flash1 = generate_data.flashdata(constants, properties, conditions1, [zs],
                                     "Vapor_Liquid")[0]

    assert np.allclose(flash1.betas, [0.18863163171737235, 0.8113683682826276])

    conditions2 = {"P": [10000.0], "H": [-10000]}
    flash2 = generate_data.flashdata(constants, properties, conditions2, [zs],
                                     "Vapor_Liquid")[0]
    assert np.allclose(flash2.betas, [0.6810565326096722, 0.31894346739032775])


def test_pure_flash():
    constants, correlations = ChemicalConstantsPackage.from_IDs(['decane'])

    conditions = {"T": [300], "H": [50802]}
    zs = [1.0]
    flash0 = generate_data.flashdata(constants, correlations, conditions, [zs],
                                     "pure")[0]

    assert np.isclose(flash0.P, 537764886.6001)


def test_collect_VL():
    constants, properties = ChemicalConstantsPackage.from_IDs(['methane', 'ethane', 'nitrogen'])
    data1 = generate_data.flashdata(constants, properties, {"T": [110.0, 120.0], "P": [1e5, 2e5]},
                                    [[.25, 0.7, .05], [.25, 0.7, .05]], "Vapor_Liquid")
    train_loader = DataLoader(data1, shuffle=False,
                              batch_size=2, collate_fn=generate_data.collate_VL)

    assert list(next(iter(train_loader))[1].shape) == [2, 8]
    assert list(next(iter(train_loader))[0].shape) == [2, 14]
    print(next(iter(train_loader))[1])

    dic = dict(T=[168.2961], P=[179736.4561])
    zs = [[0.8975495883390227, 0.06624433417431613, 0.03620607748666118]]
    data2 = generate_data.flashdata(constants, properties, dic,
                                    zs, "Vapor_Liquid")
    train_loader = DataLoader(data2, shuffle=False,
                              batch_size=2, collate_fn=generate_data.collate_VL)

    assert list(next(iter(train_loader))[1].shape)[1] == 8
    assert list(next(iter(train_loader))[0].shape)[1] == 14

    data3 = generate_data.flashdata(constants, properties, dic,
                                    zs, "Vapor_Liquid")
    train_loader = DataLoader(data3, shuffle=False,
                              batch_size=2, collate_fn=generate_data.collate_VL_NParray)

    assert isinstance(next(iter(train_loader))[1], np.ndarray)
    assert list(next(iter(train_loader))[1].shape)[1] == 8
    assert list(next(iter(train_loader))[0].shape)[1] == 14


def test_generate_good_TPZ():
    ID = ['Methane', 'Ethane', 'Propane']
    constants, properties = ChemicalConstantsPackage.from_IDs(ID)
    data = generate_data.generate_good_TPZ(5000, constants, properties, "test_good_TPZ", comment=str(ID))

    assert len(data[0]) >= 5000
    assert len(data[1]) >= 5000
    assert len(data[2]) >= 5000
    assert len(data[0]) == len(data[1])
    assert len(data[1]) == len(data[2])
