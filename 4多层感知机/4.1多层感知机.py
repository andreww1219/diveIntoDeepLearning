import torch
from matplotlib import pyplot as plt

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5)):
    """
    :param X: 自变量
    :param Y: 因变量
    :param xlabel: 自变量的名称
    :param ylabel: 因变量的名称
    :param legend: 图例
    :param xlim: X轴的取值范围
    :param ylim: Y轴的取值范围
    :param xscale: X轴的缩放方式，默认为 linear
    :param yscale: Y轴的缩放方式，默认为 linear
    :param fmts: 图线的类型，默认 '-'为实线, 'm--'为红色虚线, 'g-.'为绿色点划线, 'r:'为红色点线
    :param figsize: 整张图像的大小
    :param axes: 已有的图像，默认为 None
    :return:
    """
    # 确定图像大小
    plt.figure(figsize=figsize)
    # 确定坐标轴
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    # label为标记
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    # scale为缩放方式
    if xscale: plt.xscale(xscale)
    if yscale: plt.yscale(yscale)
    # plot为绘制图像的函数，，scale为缩放方式
    for x, y, fmt in zip(X, Y, fmts):
        plt.plot(x, y, fmt)

    # 将标记绘制图例
    if legend: plt.legend(legend)
    plt.show()
    plt.close()
# 常用激活函数
x = torch.arange(-5., 5., 0.1, requires_grad=True)
# relu
y = torch.relu(x)
y.sum().backward()
x_np = x.detach().numpy()
# plot([x_np, x_np], [y.detach().numpy(), x.grad.numpy()],
#         xlabel='x', ylabel='relu(x)',
#         figsize=(12, 12), legend=['relu(x)', 'relu\'(x)'])


# sigmoid
y = torch.sigmoid(x)
x.grad.zero_()
y.sum().backward()
plot([x_np, x_np], [y.detach().numpy(), x.grad.numpy()],
        xlabel='x', ylabel='sigmoid(x)',
        figsize=(12, 12), legend=['sigmoid(x)', 'sigmoid\'(x)'])

# tanh
y = torch.tanh(x)
x.grad.zero_()
y.sum().backward()
plot([x_np, x_np], [y.detach().numpy(), x.grad.numpy()],
        xlabel='x', ylabel='tanh(x)',
        figsize=(12, 12), legend=['tanh(x)', 'tanh\'(x)'])