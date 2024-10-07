import torch
from torch import nn


# 卷积操作定义
def conv2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y
# 卷积层定义
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, X):
        return conv2d(X, self.kernel) + self.bias


X = torch.zeros((4, 6))
conv2dLayer = Conv2D(kernel_size=(2, 2))
print(X)
print(conv2dLayer.kernel)
print(conv2dLayer(X))

# 边缘检测
X = torch.zeros((4, 8))
X[:, 2:6] = 1
K = torch.tensor([[1, -1]])
print(X, '\n', K)
print(conv2d(X.T, K.T))
