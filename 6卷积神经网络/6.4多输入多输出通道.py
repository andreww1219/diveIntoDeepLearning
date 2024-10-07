import torch

def conv2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y
# 多输入
def conv2d_multi_in(X, K):
    return sum(conv2d(x, k) for x, k in zip(X, K))
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(conv2d_multi_in(X, K))

# 多输出
def conv2d_multi_in_out(X, K):
    return torch.stack([conv2d_multi_in(X, k) for k in K], 0)

K = torch.stack((K, K + 1, K + 2), 0)

print(K.shape)

print(conv2d_multi_in_out(X, K))

# 1 * 1 卷积
def conv2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = conv2d_multi_in_out_1x1(X, K)
Y2 = conv2d_multi_in_out(X, K)
print(float(torch.abs(Y1 - Y2).sum()) < 1e-6)

