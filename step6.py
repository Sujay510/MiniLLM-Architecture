import torch
import torch.nn.functional as F
import torch.nn as nn

class Simplenetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3,2)
        self.layer2 = nn.Linear(2,1)

    def forward(self,x):
        y1 = F.relu(self.layer1(x))
        y2 = self.layer2(y1)
        return y2

model = Simplenetwork()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

x = torch.tensor([2.0,3.0,4.0])
actual = torch.tensor([3.0])

for i in range(1000):
    y = model.forward(x)
    loss = (y-actual)**2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i in range(90,101):
        print(f"Iter {i}: y={y.item():.4f}, loss={loss.item():.4f}") 