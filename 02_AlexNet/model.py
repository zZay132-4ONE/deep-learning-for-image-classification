"""
@description: AlexNet模型
@author: Zzay
@create: 2022/04/12 00:44
"""
import torch
import torch.nn as nn
import torch.utils.data.dataloader


class AlexNet(nn.Module):
    """
    AlexNet网络结构：
        Conv -> Conv -> MaxPool -> Conv -> MaxPool -> Conv
        -> Conv -> Conv -> MaxPool -> FC -> FC -> FC
    AlexNet输入图像尺寸为：
        3*224*224
    """

    def __init__(self, num_classes=1000, init_weights=False):
        """
        初始化AlexNet网络结构。
        传入参数为分类的类别数量（默认为1000），以及是否已经手动初始化权重（默认为False）。

        Conv2d默认参数：
            (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        MaxPool2d默认参数：
            (kernel_size, stride, padding, dilation)
        Linear默认参数：
            (in_features, out_features, bias, device, dtype)
        """
        super(AlexNet, self).__init__()
        # 提取特征
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(11, 11), stride=(4, 4), padding=2),  # input[3, 224, 224], output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]

            nn.Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]

            nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # output[128, 6, 6]
        )
        # 分类
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, num_classes)
        )
        # 初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        AlexNet前馈过程。
        """
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        初始化参数权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


