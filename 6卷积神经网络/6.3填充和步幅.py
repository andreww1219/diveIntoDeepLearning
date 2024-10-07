import torch
from torch import nn

def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

# 填充
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)

# 步幅
conv2d = nn.Conv2d(1, 1, kernel_size=3, stride=2)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)

# 复杂示例
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
# X.shape = (8, 8)
# 正常经过 conv2d 输出的 shape 为 （6, 4)
# 由于 padding 是 (0, 1) 那么输入的 shape 变为 (8, 10)
# 那么 stride 为 1 时输出 (6, 6)
# stride 为 (3, 4) 时，输出为 (2, 2)
print(comp_conv2d(conv2d, X).shape)