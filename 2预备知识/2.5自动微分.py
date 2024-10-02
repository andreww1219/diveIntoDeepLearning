import torch
from matplotlib import pyplot as plt
# 设置自动微分
# 方式一：定义时设置
x = torch.arange(4.0, requires_grad=True)
# 方式二：.requeres_grad(True）
# x = torch.arange(4.0)
# x.requires_grad(True)

y = 2 * torch.dot(x, x)
print(y)    # 此时，y 是一个计算图

# 可对 y 求导
y.backward()
print(x.grad)

# 默认梯度会积累，一般需要将梯度清空
x.grad.zero_()
print(x.grad)

# 非标量的反向传播
x = torch.arange(4.0, requires_grad=True)
y = x * x
y.sum().backward()
print(x.grad)
x.grad.zero_()

# 分离计算
# 没有使用分离计算
x = torch.arange(4.0, requires_grad=True)
y = x * x
z = y * x
z.sum().backward()
print(x.grad, x.grad == y)
# 使用分离计算
x = torch.arange(4.0, requires_grad=True)
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad, x.grad == u)


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
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    # label为标记
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # scale为缩放方式
    plt.xscale(xscale)
    plt.yscale(yscale)
    # plot为绘制图像的函数，，scale为缩放方式
    for x, y, fmt in zip(X, Y, fmts):
        plt.plot(x, y, fmt)

    # 将标记绘制图例
    if legend:
        plt.legend(legend)
    plt.show()
    plt.close()
# f(a) 是关于 a 得分段线性函数
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)


