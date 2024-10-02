import torch

x = torch.arange(4, dtype=torch.float32)

# 求和
print(x, x.sum())
# 非降维求和
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)
print(A.sum(axis=0))
print(A / A.sum(axis=0))    # 利用广播归一化


x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype = torch.float32)
# 点积
print(x, y, torch.dot(x, y))
# 矩阵-向量积
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(torch.mv(A, x))
# 矩阵乘法
B = torch.ones(4, 3)
print(torch.mm(A, B))

# 向量的 L2-范数 及 矩阵的 F-范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
A = torch.ones((4, 9))
print(torch.norm(A))
# L1-范数
print(torch.abs(u).sum())
