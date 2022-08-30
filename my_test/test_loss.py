import sys
sys.path.append("..")
from model import my_loss_function
import torch
import numpy as np



def test_MB_Loss():
    criterian = my_loss_function.My_Mass_Balance_loss(1, 1)
    y_pred = torch.tensor([[0.2, 0.2, 0.6, 0.1, 0.1, 0.1, 0.9, 0.1]], requires_grad=True)
    y_real = torch.tensor([[0.1, 0.3, 0.6, 0, 0, 0, 1, 0]])
    original_fraction = torch.tensor([[0.1, 0.3, 0.6]])
    loss = criterian(y_pred, y_real, original_fraction)


    gas = np.array([0.2, 0.2, 0.6])
    liquid = np.array([0.1, 0.1, 0.1])
    mass = torch.tensor(gas * 0.9 + liquid * 0.1,requires_grad=True,dtype=torch.float32).unsqueeze(0)

    loss_c = torch.nn.MSELoss()
    loss_correct=loss_c(mass, torch.Tensor([[0.1, 0.3, 0.6]]))+loss_c(y_pred,y_real)

    assert loss_correct==loss


    y_pred = torch.tensor([[0.2, 0.2, 0.6, 0.1, 0.1, 0.1, 0.9, 0.1], [0.2, 0.2, 0.6, 0.1, 0.1, 0.1, 0.9, 0.2]],
                          requires_grad=True)
    y_real = torch.tensor([[0.1, 0.3, 0.6, 0, 0, 0, 1, 0], [0.1, 0.3, 0.6, 0, 0, 0, 1, 0]])
    original_fraction = torch.tensor([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]])
    loss = criterian(y_pred, y_real, original_fraction)

    assert np.isclose(loss.item(), 0.01800833)


