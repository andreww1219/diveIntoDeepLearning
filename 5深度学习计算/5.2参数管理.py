import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

# 参数访问
print(net[2].weight)
print(net[2].bias)
# 只访问参数的值，即 requires_grad = False
print(net[2].bias.data)
# 访问包含所有参数的字典
print(net[2].state_dict())

print()
print([(name, param.shape) for name, param in net[0].named_parameters()])
print([(name, param.shape) for name, param in net.named_parameters()])
print([param.shape for param in net.parameters()])

# 嵌套块
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))

print(rgnet)
print([(name, param.shape) for name, param in rgnet[0][1][0].named_parameters()])

# 参数共享
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])


# 延后初始化

net = nn.Sequential(
        nn.LazyLinear(256),
        nn.ReLU(),
        nn.LazyLinear(10)
)
X = torch.rand((5, 20))
print(net(X).shape)

# 自定义层

class MeanLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean(axis=1)
X = torch.rand((1, 5))
print(X)
layer = MeanLayer()
print(layer(X))

