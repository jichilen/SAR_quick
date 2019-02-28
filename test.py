import torch
from torchsummary import summary
from torch import nn
class A(nn.Module):
    def __init__(self):
        super(A,self).__init__()
        self.linear=nn.Linear(10,5)
        self.b=B()
    def forward(self,x):
        x=self.linear(x)
        x,y,z=self.b(x)
        return x

class B(nn.Module):
    def __init__(self):
        super(B,self).__init__()
    def forward(self,x):
        return x,x,x

def main():
    a=A()
    a=a.cuda()
    summary(a,(2,10))

if __name__ == '__main__':
    main()
