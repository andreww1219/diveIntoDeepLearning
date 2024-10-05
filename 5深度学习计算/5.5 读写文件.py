import torch
from torch import nn
from torch.nn import functional as F

# 保存/读取张量
x = torch.arange(4)
torch.save(x, 'x-file')
y = torch.load('x-file')
print(x, y)

# 保存/读取字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict, mydict2)

# 保存/读取模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
net = MLP()
torch.save(net.state_dict(), 'mlp_state_dict')

clone = MLP()
clone.load_state_dict(torch.load('mlp_state_dict'))

X = torch.rand((1, 20))
print(net(X) == clone(X))
