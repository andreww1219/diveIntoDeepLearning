
import torch
from matplotlib import pyplot as plt
import random

def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 观察第二个维度与标签的关系
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy())
plt.show()


# 小批量
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))     # 样本下标
    random.shuffle(indices)                 # 打乱顺序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 5
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break



# 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# 定义模型
def linear_reg(w, b, X):
    return torch.matmul(X, w) + b
# 定义损失函数
def squared_loss(y_hat, y):
    return (y - y_hat.reshape(y.shape)) ** 2 / 2
# 定义优化算法
def sgd(params, lr, batch_size):
    """
    小批量梯度下降
    :param params: 参数
    :param lr: 学习率
    """
    with torch.no_grad():
        # with torch.no_grad(): 以内的空间计算结果得 requires_grad 为 False
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
# 训练
lr = 0.01
num_epochs = 3
net = linear_reg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(w, b, X), y)  # X和y的小批量损失
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(w, b, features), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
