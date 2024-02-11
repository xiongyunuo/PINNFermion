import torch
from torch import nn
from torch import autograd
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

xit=[]
es=[]

file=open('datafermi5454.txt','r')
for line in file.readlines():
    y=line.split()
    if len(y)==3:
        xit.append([float(y[0]),float(y[1])])
        es.append([float(y[2])/200.0])
file.close()
#print(xit)
#print(es)

device="cpu"

class nnActi(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)

class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.acti=nnActi()
        self.f2=nn.Linear(2,50)
        self.f3=nn.Linear(50,50)
        self.f4=nn.Linear(50,50)
        self.f5=nn.Linear(50,50)
        self.f6=nn.Linear(50,50)
        self.f7=nn.Linear(50,1)
    def forward(self,x):
        out=self.f2(x)
        out=self.acti(out)
        out=self.f3(out)
        out=self.acti(out)
        out=self.f4(out)
        out=self.acti(out)
        out=self.f5(out)
        out=self.acti(out)
        out=self.f6(out)
        out=self.acti(out)
        out=self.f7(out)
        return out

model=FNN().to(device)
train_data2=torch.tensor(xit)
train_out2=torch.tensor(es)
train_number=len(es)
#print(train_number)
#print(train_data)
#print(train_out)
batch_size=20
cons_input=torch.tensor([[-2.0,0.0],[-1.9,0.0],[-1.8,0.0],[-1.7,0.0],[-1.6,0.0],[-1.5,0.0],[-1.4,0.0],[-1.3,0.0],[-1.25,0],[-1.2,0.0],[-1.15,0],[-1.1,0.0],[-1.05,0.0],[-1.0,0.0],[-0.95,0.0],[-0.9,0.0],[-0.85,0],[-0.8,0.0],[-0.7,0.0],[-0.6,0.0],[-0.5,0.0],[-0.4,0.0],[-0.3,0.0],[-0.2,0.0],[-0.1,0.0],[0.0,0.0],[0.1,0.0],[0.2,0.0],[0.3,0.0],[0.4,0.0],[0.5,0.0]])
def dfx(xy,f):
    grad = autograd.grad([f],[xy],grad_outputs=torch.ones(f.shape,dtype=torch.float).to(device),create_graph=True)[0]
    n=grad.size()[0]
    return (grad[:,1]).reshape((n,1))
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
def training():
    count=0
    train_data=torch.clone(train_data2)
    train_out=torch.clone(train_out2)
    idx = np.random.permutation(train_number)
    train_data=train_data[idx]
    train_out=train_out[idx]
    while count<train_number:
        input=train_data[count:min(count+batch_size,train_number)]
        output=train_out[count:min(count+batch_size,train_number)]
        pred=model(input)
        #print(input)
        #print(output)
        #print(pred)
        #exit()
        cons_input.requires_grad=True
        pred2=model(cons_input)
        #print(dfx(cons_input,pred2))
        #exit()
        loss=(output-pred).pow(2).mean()
        loss+=1.0*dfx(cons_input,pred2).pow(2).mean()#PINN
        loss.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()
        count+=batch_size
    print(loss)
for i in range(0,5000):
    training()
input=train_data2[0:1]
output=train_out2[0:1]
pred=model(input)
print(input)
print(output)
print(pred)
cons_input.requires_grad=True
pred2=model(cons_input)
print(dfx(cons_input,pred2).pow(2).mean())
print(pred2[0])
px,py=100,100
points=torch.zeros((px*py,2))
for i in range(0,px):
    for j in range(0,py):
        points[i*py+j,0]=-1+2.0*i/(px-1)
        points[i*py+j,1]=-0.25+5*j/(py-1)
count=5
outfile=open('outfermi5454.txt','w')
while count<px*py:
    input=points[count:(count+batch_size)]
    pred=model(input)
    for i in range(0,input.size()[0]):
        outfile.write(str(input[i,0].item())+' '+str(input[i,1].item())+' '+str(pred[i,0].item())+'\n')
    count+=batch_size
outfile.close()
