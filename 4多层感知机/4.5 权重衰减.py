import torch
from matplotlib import pyplot as plt
from torch import nn
import random

# 生成数据集 初始化参数
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, w.shape[0]))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))     # 样本下标
    random.shuffle(indices)                 # 打乱顺序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 4, -3.4
train_data = synthetic_data(true_w, true_b, n_train)
train_iter = data_iter(batch_size, *train_data)
test_data = synthetic_data(true_w, true_b, n_test)
test_iter = data_iter(batch_size, *test_data)

plt.scatter(train_data[0][:, 1].detach().numpy(), train_data[1].detach().numpy())
plt.show()

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定义损失函数
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
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
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: linear_reg(w, b, X), squared_loss
    num_epochs, lr = 100, 0.03
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        print('loss：', loss(net(train_data[0]), train_data[1]).sum())
        print('w的L2范数是：', torch.norm(w).item())
        with torch.no_grad():
            train_l = loss(net(train_data[0]), train_data[1])
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
    print('w的L2范数：', net[0].weight.norm().item())
train_concise(0.01)