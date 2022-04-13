"""
@description: 调用AlexNet模型训练
@author: Zzay
@create: 2022/04/12 20:11
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader
from torchvision import transforms, datasets, utils
from tqdm import tqdm
from model import AlexNet


def main():
    # 设置运算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 预处理函数
    data_transform = {
        # 训练用
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        # 测试用
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    # 设置数据集路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../00_dataset"))
    image_path = os.path.join(data_root, "flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # 获取训练数据集
    train_set = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
    train_num = len(train_set)
    # swap the positions of inputs and labels, and write into a json file, for later processing
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_set.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    # set parameters like batch_size, num_workers
    batch_size = 32
    num_workers = 0
    # set train loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

    # 获取验证数据集
    test_set = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                    transform=data_transform["val"])
    test_num = len(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                              shuffle=False, num_workers=num_workers)
    print("using {} images for training, {} images for testing.".format(train_num, test_num))

    # 建立模型，构造损失函数和优化器
    net = AlexNet(num_classes=5, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)

    # 训练过程
    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        # test
        acc = 0.0
        net.eval()  # 停用Dropout等处理
        with torch.no_grad():  # 不再需要持久记录计算图
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                inputs, labels = data
                outputs = net(inputs.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, labels.to(device)).sum().item()
        test_acc = acc / test_num
        print('[epoch %d] test_loss: %.3f  test_acc: %.3f' % (epoch + 1, running_loss / train_steps, test_acc))
        # 若在测试中获得更高的accuracy，则记录并存下当前的权重数据
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), save_path)
    print('Finished Training')


if __name__ == '__main__':
    main()
