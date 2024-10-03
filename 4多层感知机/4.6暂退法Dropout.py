import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils import data
def dropout_layer(X, dropout):
    """
    按 dropout 比例丢弃元素，并进行缩放（除以 1 - dropout）
    """
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = torch.rand(X.shape) > dropout    # torch.rand 生成 [0, 1)的均匀分布的随机数
    return mask.float() * X / (1 - dropout)

X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))



num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)  # 求和后保留被求和的维度
    return X_exp / partition
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);


# 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 定义损失函数
def cross_entropy(y_hat, y):
    y_hat = softmax(y_hat)
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
loss = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    for X, y in train_iter:
        l = loss(net.forward(X), y)  # X和y的小批量损失
        l.sum().backward()
        sgd(net.parameters(), lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        acc = evaluate_accuracy(net, test_iter)
        print(f'epoch {epoch + 1}, acc {acc:f}')