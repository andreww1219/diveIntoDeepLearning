import torch
from torch import nn

def pool2d(X, pool_size, mode='max'):
    h, w = pool_size
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + h, j: j + w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + h, j: j + w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# 最大池化
print(pool2d(X, (2, 2)))
# 平均池化
print(pool2d(X, (2, 2), mode='avg'))



# 填充和步幅
X = torch.arange(9.).reshape(1, 1, 3, 3)
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
# ((3, 3) + 2 * (0, 1) - (2, 3)) / (2, 3) + （1, 1) = (1, 1)
print(pool2d(X))

# 多个通道
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
X = torch.cat((X, X + 1), 1)
print(X.shape)
print(pool2d(X).shape)
