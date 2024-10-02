import torch

# 创建行向量
x = torch.arange(12)
print(x)

# 元素的数量
print(x.numel())
# 张量的形状
print(x.shape)
# 更改形状
print(x.reshape(3, 4))

# 全零张量
print(torch.zeros((2, 3, 4)))
# 全一张量
print(torch.ones((2, 3, 4)))
# 随机数张量
print(torch.rand((2, 3, 4)))
# 创建时初始化
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
# 四则运算
print(x + y, x - y, x * y, x / y, x ** y)
# 自然指数
print(torch.exp(x))
# 求和
print(torch.sum(x))
# 逻辑运算
print(x == y)

# 张量连接
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 按行 (3, 4) 和 (3, 4) 连接成 (6, 4)
print(torch.cat((X, Y), dim=0))
# 按列 (3, 4) 和 (3, 4) 连接成 (3, 8)
print(torch.cat((X, Y), dim=1))

# 广播机制
# 简单版
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a + b, '\n')
# 复杂版
# 从尾部维度数起，d最后一个维度为1，c与d倒数第二个维度相等，d倒数第三个维度不存在。故可广播
c = torch.arange(12).reshape((2, 3, 2))
d = torch.arange(3).reshape((3, 1))
print(c)
print(d)
print(c + d)

# 索引和切片，同 numpy
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(X)
# 利用切片取值
print(X[-1])    # 相当于 X[X.shape[0] - 1]
print(X[1: 3])  # 左闭右开 X[a, b] 相当于 [a, b)
# 利用切片赋值
X[1, 2] = 9
print(X)
X[0:2, :] = 12
print(X)

# 浪费内存的写法
X = X + Y
# 节省内存的写法
X += Y
X[:] = X + Y

A = X.numpy()
# ndarray 转 tensor
B = torch.tensor(A)
print(type(A), type(B))
# 使用 .item() 取单个元素为 Python 基本类型元素
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
