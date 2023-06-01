import torch
from torch.autograd import Variable

x_data = torch.tensor([ [1.0],[2.0],[3.0] ])
y_data = torch.tensor([ [2.0],[4.0],[6.0] ])

class MyModel(torch.nn.Module): #torch.nn.Model 상속 받음
    def __init__(self):
        super(MyModel, self).__init__() #처음 initialize 할 때, 부모도 init
        
        self.linear = torch.nn.Linear(1,1)
        
    def forword(self,x):
        y_pred = self.linear(x)
        return y_pred

model = MyModel()

criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #optimizer 안에서 gradient 계산과 backprop이 발생한다.
 #model.parameters()를 통해 gradient update, 그 중에서 SGD 방법을 서서 optimize, learning rate 지정

#@ Train Start
for epoech in range(500):
    y_pred = model(x_data)
    
    loss = criterion(y_pred, y_data) #loss 계산
    
    optimizer.zero_grad() #gradient descent 직전에 초기화
    
    loss.backward() #구한 loss에서 gradient 구함
    
    optimizer.step()