import os.path
import pandas as pd
import torch

data_file = os.path.join('..', 'datas', 'heart', 'heart.csv')
data = pd.read_csv(data_file)
print(data.head())

x, y = data.iloc[:, :-2], data.iloc[:, -1]
print(x.head())
print(y.head())
# 需要先转换为 ndarray 再转换为 tensor
X = torch.tensor(x.to_numpy(dtype=float))
Y = torch.tensor(y.to_numpy(dtype=float))
print(type(X), X.shape)
print(type(Y), Y.shape)