"""
@description: 调用LeNet模型训练
@author: Zzay
@create: 2022/04/11 01:26
"""

import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import LeNet


def main():
    # 获取预处理函数
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 获取训练数据集（50000张图片）
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)
    # 获取测试数据集（10000张图片）
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000,
                                              shuffle=False, num_workers=0)
    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()

    # 建立模型，损失函数，优化器
    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)

    # 训练过程
    for epoch in range(5):
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get a pair of input data and label
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # predict using the model
            outputs = net(inputs)
            # compute loss of current prediction
            loss = loss_function(outputs, labels)
            # loss backward
            loss.backward()
            # optimize params
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(test_image)
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, test_label).sum().item() / test_label.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0
    print('Finished Training')

    # 保存模型的参数权重
    save_path = './LeNet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
