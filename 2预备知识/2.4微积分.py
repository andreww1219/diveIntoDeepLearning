from matplotlib import pyplot as plt
import numpy as np


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
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
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
    plt.legend(legend)
    plt.show()
    plt.close()


"""
练习一
绘制函数 y = f(x) = x^3 - 1/x 及其在 x = 1 处切线的图像

f(1) = 0
f'(x) = 3x^2 + 1/x^2    f'(1) = 3
那么，x = 1 处切线方程为 y = 3x - 3
"""


def f(x):
    return x**3 - 1/x


x = np.arange(0.1, 3, 0.1)
print(x)
print(f(x))
print(3 * x - 3)
plot(X=[x, x], Y=[f(x), 3 * x - 3],
     xlabel='x', ylabel='f(x)',
     legend=['f(x)', 'Tangent Line(x=1)'])
