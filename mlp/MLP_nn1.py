import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# 1. 加载MNIST数据集，并返回训练集和测试集的数据加载器
def load_data_mnist(batch_size): # batch_size (int): 每个批次的样本数量
    transform = transforms.Compose([
        transforms.ToTensor(), # 将PIL图像转换为PyTorch张量
        transforms.Normalize((0.1307,), (0.3081,))  #  # MNIST的标准归一化参数(均值=0.1307, 标准差=0.3081)
    ])
    # 定义数据预处理转换
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform) #四个参数的含义：数据存储路径，加载训练集，如果不存在则下载，应用定义的数据转换
    test_data = datasets.MNIST('../data', train=False, download=True, transform=transform) #四个参数的含义：数据存储路径，加载测试集，如果不存在则下载，应用定义的数据转换
    print("数据集加载完成！")
    return (
        DataLoader(train_data, batch_size, shuffle=True, num_workers=2), # 训练集数据加载器(打乱顺序，使用2个工作进程)
        DataLoader(test_data, batch_size, shuffle=False, num_workers=2)  # 测试集数据加载器(不打乱顺序，使用2个工作进程)
    )

# 2. 模型定义 简单的两层神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, num_inputs=784, num_hiddens=256, num_outputs=10): #参数分别为：输入层特征维度，默认为784(MNIST图像展平后的尺寸28x28)；隐藏层神经元数量，默认为256；输出层维度，默认为10(对应MNIST的10个类别)
        super().__init__()
        #第一层参数: 输入层到隐藏层的权重和偏置
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        #第二层参数: 隐藏层到输出层的权重和偏置
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))

    """定义前向传播过程"""
    def forward(self, X):
        X = X.reshape(-1, 784)  # # 将输入图像展平为(batch_size, 784)的形状
        H = torch.relu(X @ self.W1 + self.b1)  # 第一层计算: 线性变换后通过ReLU激活函数
        return H @ self.W2 + self.b2 # 第二层计算: 线性变换(无激活函数，输出logits)


# 3. 训练神经网络模型 并评估其性能
def train_model(net, train_iter, test_iter, num_epochs=10, lr=0.1):
    # 初始化优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr) # 使用随机梯度下降优化器
    criterion = nn.CrossEntropyLoss() # 使用交叉熵损失函数

    train_loss_list = [] # 记录每个epoch的训练损失
    train_acc_list = []  # 记录每个epoch的训练准确率
    test_acc_list = []   # 记录每个epoch的测试准确率

    print("开始训练...")
    for epoch in range(num_epochs):
        total_loss, total_correct, total_samples = 0, 0, 0 # 初始化统计变量

        # 训练阶段
        for X, y in train_iter:
            optimizer.zero_grad() #清空梯度缓存
            output = net(X)       # 计算模型输出
            loss = criterion(output, y) # 计算损失
            loss.backward()             # 计算梯度
            optimizer.step()            # 更新参数

            total_loss += loss.item() # 累计损失
            total_correct += (output.argmax(dim=1) == y).sum().item() # 累计正确预测数
            total_samples += y.size(0)  # 累计样本数

        # 计算训练指标
        train_loss = total_loss / len(train_iter) # 计算平均训练损失
        train_acc = total_correct / total_samples # 计算训练准确率

        # 测试阶段
        test_correct, test_samples = 0, 0
        with torch.no_grad(): # 禁用梯度计算
            for X, y in test_iter:
                output = net(X)
                test_correct += (output.argmax(dim=1) == y).sum().item()
                test_samples += y.size(0)
        test_acc = test_correct / test_samples

        # 记录指标
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.3f}, '
              f'Test Acc: {test_acc:.3f}')

    """可视化结果"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Train Acc')
    plt.plot(test_acc_list, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 设置随机种子保证可重复性
    torch.manual_seed(42)

    # 参数设置
    batch_size = 256
    num_epochs = 10
    learning_rate = 0.1

    # 加载数据
    train_iter, test_iter = load_data_mnist(batch_size)

    # 初始化模型
    model = SimpleNN()

    # 开始训练
    train_model(model, train_iter, test_iter, num_epochs, learning_rate)