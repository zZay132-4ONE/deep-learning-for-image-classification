"""
@description: GoogLeNet模型
@author: Zzay
@create: 2022/04/13 15:16
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    """
    GoogLeNet网络结构：
        Conv7 -> MaxPool -> LRN
        -> Conv1 -> Conv3 -> LRN -> MaxPool
        -> Inception(3a) -> Inception(3b) -> MaxPool
        -> Inception(4a) -> Inception(4b) & Aux(1)
        -> Inception(4c) -> Inception(4d) -> Inception(4e) & Aux(2) -> MaxPool
        -> Inception(5a) -> Inception(5b)
        -> AvgPool -> Dropout(0.4) -> FC-1024 -> Softmax-1000
    VGGNet输入图像尺寸为：
        3 x 224 x 224
    VGGNet输出为：
        1000个类别的概率
    """

    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        """
        初始化GoogLeNet网络结构。

        传入参数：
            num_classes： 分类的类别数量（默认为1000）
            aux_logits：  是否需要辅助分类器（默认需要）
            init_weights：是否已经手动初始化权重（默认为False）
        """
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        # conv, maxPool
        self.conv1 = BasicConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # conv, maxPool
        self.conv2 = BasicConv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = BasicConv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # inception(3a), inception(3b), maxPool
        self.inception3a = Inception(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128,
                                     ch5x5red=16, ch5x5=32, pool_proj=32)
        self.inception3b = Inception(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192,
                                     ch5x5red=32, ch5x5=96, pool_proj=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # inception (4a), inception (4b), inception (4c), inception (4d), inception (4e), maxPool
        self.inception4a = Inception(in_channels=480, ch1x1=192, ch3x3red=96, ch3x3=208,
                                     ch5x5red=16, ch5x5=48, pool_proj=64)
        self.inception4b = Inception(in_channels=512, ch1x1=160, ch3x3red=112, ch3x3=224,
                                     ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4c = Inception(in_channels=512, ch1x1=128, ch3x3red=128, ch3x3=256,
                                     ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4d = Inception(in_channels=512, ch1x1=112, ch3x3red=144, ch3x3=288,
                                     ch5x5red=32, ch5x5=64, pool_proj=64)
        self.inception4e = Inception(in_channels=528, ch1x1=256, ch3x3red=160, ch3x3=320,
                                     ch5x5red=32, ch5x5=128, pool_proj=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # inception (5a), inception (5b)
        self.inception5a = Inception(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320,
                                     ch5x5red=32, ch5x5=128, pool_proj=128)
        self.inception5b = Inception(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384,
                                     ch5x5red=48, ch5x5=128, pool_proj=128)

        # auxiliary classifiers
        if self.aux_logits:
            self.aux1 = InceptionAux(in_channels=512, num_classes=num_classes)
            self.aux2 = InceptionAux(in_channels=528, num_classes=num_classes)

        # avgPool, dropout, fc
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        GoogLeNet内部前馈过程。
        """
        # [N x 3 x 224 x 224] -> [N x 64 x 112 x 112]
        x = self.conv1(x)
        # [N x 64 x 112 x 112] -> [N x 64 x 56 x 56]
        x = self.maxpool1(x)

        # [N x 64 x 56 x 56] -> [N x 64 x 56 x 56]
        x = self.conv2(x)
        # [N x 64 x 56 x 56] -> [N x 192 x 56 x 56]
        x = self.conv3(x)
        # [N x 192 x 56 x 56] -> [N x 192 x 28 x 28]
        x = self.maxpool2(x)

        # [N x 192 x 28 x 28] -> [N x 256 x 28 x 28]
        x = self.inception3a(x)
        # [N x 256 x 28 x 28] -> [N x 480 x 28 x 28]
        x = self.inception3b(x)
        # [N x 480 x 28 x 28] -> [N x 480 x 14 x 14]
        x = self.maxpool3(x)

        # [N x 480 x 14 x 14] -> [N x 512 x 14 x 14]
        x = self.inception4a(x)

        if self.training and self.aux_logits:
            # eval model omit this layer
            aux1 = self.aux1(x)

            # [N x 512 x 14 x 14] -> [N x 512 x 14 x 14]
        x = self.inception4b(x)
        # [N x 512 x 14 x 14] -> [N x 512 x 14 x 14]
        x = self.inception4c(x)
        # [N x 512 x 14 x 14] -> [N x 528 x 14 x 14]
        x = self.inception4d(x)

        if self.training and self.aux_logits:
            # eval model omit this layer
            aux2 = self.aux2(x)

        # [N x 528 x 14 x 14] -> [N x 832 x 14 x 14]
        x = self.inception4e(x)
        # [N x 832 x 14 x 14] -> [N x 832 x 7 x 7]
        x = self.maxpool4(x)

        # [N x 832 x 7 x 7] -> [N x 832 x 7 x 7]
        x = self.inception5a(x)
        # [N x 832 x 7 x 7] -> [N x 1024 x 7 x 7]
        x = self.inception5b(x)
        # [N x 1024 x 7 x 7] -> [N x 1024 x 1 x 1]
        x = self.avgpool(x)

        # [N x 1024 x 7 x 7] -> N x 1024
        x = torch.flatten(x, start_dim=1)
        # N x 1024 -> N x 1024
        x = self.dropout(x)
        # N x 1000 (num_classes)
        x = self.fc(x)

        # auxiliary classifiers
        if self.training and self.aux_logits:
            # eval model omit this layer
            return x, aux2, aux1

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


class Inception(nn.Module):
    """
    Inception结构。
        branch1：Conv1；
        branch2：Conv1 -> Conv3；
        branch3：Conv1 -> Conv5；
        branch4：Maxpool3 -> Conv1 ；
    """

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5, ch5x5red, pool_proj):
        """
        初始化，需要传入输入通道数、输出通道数等参数。
        """
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=(1, 1)),
            BasicConv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=(3, 3), padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=(1, 1)),
            BasicConv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=(5, 5), padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=(1, 1))
        )

    def forward(self, x):
        """
        Inception内部前馈过程：获取并拼接4个branch的输出。
        """
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    """
    辅助分类器Inception。
    """

    def __init__(self, in_channels, num_classes):
        """
        初始化，需要传入输入通道数、分类类别数。
        """
        super(InceptionAux, self).__init__()
        self.avgPool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=(1, 1))

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        """
        辅助分类器Inception内部前馈过程。
        """
        # extract features
        # input:  {aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14}
        # output: {aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4}
        x = self.avgPool(x)
        # input:  {aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4}
        # output: N x 128 x 4 x 4
        x = self.conv(x)

        # classify
        # input: N x 128 x 4 x 4, output: N x 2048
        x = torch.flatten(x, start_dim=1)
        x = F.dropout(x, p=0.5, training=self.training)
        # input: N x 2048, output: N x 1024
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, p=0.5, training=self.training)
        # input: N x 1024, output: N x num_classes
        x = self.fc2(x)
        return x


class BasicConv2d(nn.Module):
    """
    封装卷积层和ReLU激活层。
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        """
        初始化，需要传入输入通道数、输出通道数等参数。
        """
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        “卷积+relu”前馈过程。
        """
        x = self.conv(x)
        x = self.relu(x)
        return x
