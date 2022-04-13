"""
@description: VGGNet模型
@author: Zzay
@create: 2022/04/13 13:00
"""
import torch
import torch.nn as nn

# 官方预训练参数权重文件
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

# VGG网络结构配置选择
configs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def make_features(config: list):
    """
    构造网络结构中的特征提取层
    """
    in_channels = 3
    layers = []
    for param in config:
        if param == "M":
            # MaxPool
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Conv2d + ReLU
            conv2d = nn.Conv2d(in_channels, param, kernel_size=(3, 3), stride=(1, 1), padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = param
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """
    VGGNet网络结构（以Config D为例）：
        Conv3 -> Conv3 -> MaxPool
        -> Conv3 -> Conv3 -> MaxPool
        -> Conv3 -> Conv3 -> Conv3 -> MaxPool
        -> Conv3 -> Conv3 -> Conv3 -> MaxPool
        -> Conv3 -> Conv3 -> Conv3 -> MaxPool
        -> FC-4096 -> Dropout(0.5) -> FC-4096 -> Dropout(0.5) -> FC-1000 -> Softmax
    VGGNet输入图像尺寸为：
        3 x 224 x 224
    VGGNet输出为：
        1000个类别的概率
    """

    def __init__(self, features, num_classes=1000, init_weights=False):
        """
        初始化VGGNet网络结构。

        传入参数：
            features：    网络的特征提取层结构
            num_classes： 分类的类别数量（默认为1000）
            init_weights：是否已经手动初始化权重（默认为False）
        """
        super(VGG, self).__init__()
        # 提取特征
        self.features = features
        # 分类
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        VGGNet前馈过程。
        """
        # input: [N, 3, 224, 224], output: [N, 512, 7, 7]
        x = self.features(x)
        # input: [N, 512, 7, 7], output: N x 512 * 7 * 7
        x = torch.flatten(x, start_dim=1)
        # input: N x 512 * 7 * 7, output: N x 1000
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        初始化卷积层和全连接层的参数权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

def vgg(model_name="vgg16", **kwargs):
    """
    构建VGGNet模型。
    传入参数：
        model_name：选用的VGGNet配置名
    """
    assert model_name in configs, "Warning: model config {} is not valid!".format(model_name)
    config = configs[model_name]
    model = VGG(make_features(config), **kwargs)
    return model
