import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from torch.utils.data import TensorDataset, DataLoader


# 生成模拟数据
def generate_data(n_samples=1000, n_features=10):
    X_train, y_train = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    X_test, y_test = make_classification(n_samples=n_samples, n_features=n_features, random_state=43, flip_y=0.2)
    return X_train, y_train, X_test, y_test


# 估计密度比
def estimate_density_ratio(X_train, X_test, bandwidth=1.0):
    kde_train = KernelDensity(bandwidth=bandwidth).fit(X_train)
    kde_test = KernelDensity(bandwidth=bandwidth).fit(X_test)

    log_density_train = kde_train.score_samples(X_train)
    log_density_test = kde_test.score_samples(X_train)

    density_ratio = np.exp(log_density_test - log_density_train)
    return density_ratio


# 定义多层感知机
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for inputs, targets, weights in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            weighted_loss = (loss * weights).mean()
            weighted_loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {weighted_loss.item()}')


# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


# 主函数
def main():
    # 生成数据
    X_train, y_train, X_test, y_test = generate_data()

    # 估计密度比
    density_ratio = estimate_density_ratio(X_train, X_test)

    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    weights_tensor = torch.tensor(density_ratio, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, weights_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 定义模型、损失函数和优化器
    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = 2
    model = Net(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, epochs=100)

    # 评估模型
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()