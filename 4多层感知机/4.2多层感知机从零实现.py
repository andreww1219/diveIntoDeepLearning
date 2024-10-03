import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils import data

# 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
# 定义模型
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)  # 求和后保留被求和的维度
    return X_exp / partition
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return softmax(H @ W2 + b2)
# 定义损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)        # 得到每一行最大的下标
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())   # 预测正确返回1，否则返回0
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
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
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))


batch_size = 18
train_iter, test_iter = load_data_fashion_mnist(batch_size)

lr = 0.03
num_epochs = 10
loss = cross_entropy
for epoch in range(num_epochs):
    for X, y in train_iter:
        l = loss(net(X), y)  # X和y的小批量损失
        l.sum().backward()
        sgd(params, lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        acc = evaluate_accuracy(net, test_iter)
        print(f'epoch {epoch + 1}, acc {acc:f}')
