from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, PRMIX, \
    FlashVL
from data import generate_data
import numpy as np
constants, properties = ChemicalConstantsPackage.from_IDs(['methane', 'ethane', 'nitrogen'])
print(dir(properties))
a=generate_data.flashdata(constants,properties,{"T":list(np.linspace(100,300,100)),"P":[1e5 for i in range(100)]},[[.25, 0.75, .05] for i in range(100)],"Vapor_Liquid")
data_x=[]
data_y=[]
for i in range(100):
    print(a[i].liquids)
    try:
        data_y.append(a[i].gas.zs[2])
        data_x.append(i)
    except:
        pass

print(data_y)
import matplotlib.pyplot as plt
plt.plot(data_x,data_y)
plt.show()