import sys
sys.path.append("..")
from model import ArtificialNN,model_stack
import torch
import numpy as np
import xgboost
import lightgbm
from sklearn.multioutput import MultiOutputRegressor




def test_stack_model():
    num = 200
    data = np.ones((num, 14)) + np.random.randn(num, 14) * 10
    target = np.ones((num, 10)) + np.random.randn(num, 10) * 0.1

    stak_model = model_stack.stacking_model("test")
    model1 = xgboost.XGBRegressor()
    model2 = MultiOutputRegressor(lightgbm.LGBMRegressor())
    model3 = ArtificialNN.Neural_Model_Sklearn_style(ArtificialNN.ANN, {"input_num": 10, "output_num": 10})

    stak_model.add_model(model1)
    stak_model.add_model(model2)
    stak_model.add_model(model3)
    stak_model.fit(data, target)


    print(stak_model.score(data, target))
    print(stak_model.predict(data))


test_stack_model()