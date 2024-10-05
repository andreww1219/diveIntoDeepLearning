import torch
from torch import nn

# GPU 使用情况

print(torch.cuda.is_available())

print(torch.cuda.device_count())

# 指定张量的环境

x = torch.tensor([1, 2, 3])
print(x.device)

def try_gpu(i=0):
    if i < torch.cuda.device_count():
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

X = torch.ones(2, 3, device=try_gpu())
print(X.device)

# 复制张量到不同设备

X = torch.ones(2, 3, device=try_gpu())
Y = torch.rand(2, 3, device=try_gpu(1))
# Z = X.cuda(1)

# 神经网络与 GPU

net = nn.Sequential(nn.Linear(3, 1))
# net = net.to(device=try_gpu(1))

print(net(Y))
print(net[0].weight.device)