import torch
torch.cuda.is_available()
def test(a,b,c):
    print(a,b,c)

a=1;
aw={"b":0,"c":2}
test(a,**aw)