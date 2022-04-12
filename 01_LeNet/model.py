"""
@description: LeNet模型
@author: Zzay
@create: 2022/04/11 01:00
"""
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    LeNet网络结构：
        Conv -> MaxPool -> Conv -> MaxPool -> FC -> FC -> FC
    LeNet输入图像尺寸为：
        3 x 32 x 32
    LeNet输出为：
        10个类别的概率
    """

    def __init__(self):
        """
        初始化LeNet网络结构。

        Conv2d默认参数：
            (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        MaxPool2d默认参数：
            (kernel_size, stride, padding, dilation)
        Linear默认参数：
            (in_features, out_features, bias, device, dtype)
        """
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        LeNet前馈过程。
        """
        x = F.relu(self.conv1(x))       # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)               # output(16, 14, 14)
        x = F.relu((self.conv2(x)))     # output(32, 10, 10)
        x = self.pool2(x)               # output(32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)      # output(32 * 5 * 5)
        x = F.relu(self.fc1(x))         # output(120)
        x = F.relu(self.fc2(x))         # output(84)
        x = self.fc3(x)                 # output(10)
        return x
