from itertools import chain
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
import numpy as np
a=np.array([[-0.1,1.1]])
b=np.array([[0,1]])
print(mean_squared_error(a,b))