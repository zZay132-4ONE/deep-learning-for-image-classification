"""
@description: 训练GoogLeNet模型
@author: Zzay
@create: 2022/04/13 17:30
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader
from tqdm import tqdm
from torchvision import transforms, datasets
from model import GoogLeNet


def main():
    # 设置运算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 预处理函数
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # 设置相关训练参数
    batch_size = 32
    num_workers = 0

    # 获取数据集路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../00_dataset"))
    image_path = os.path.join(data_root, "flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # 获取数据集
    # train
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    # test
    test_set = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                    transform=data_transform["val"])
    test_num = len(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    print("using {} images for training, {} images for testing.".format(train_num, test_num))

    # 建立模型，构造损失函数和优化器
    net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    # 训练模型
    epochs = 30
    best_acc = 0.0
    save_path = './googleNet.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            inputs, labels = data
            optimizer.zero_grad()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            logits, aux_logits2, aux_logits1 = net(inputs.to(device))
            # 计算最终分类，以及辅助分类的损失
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        # test
        net.eval()
        acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in test_bar:
                inputs, labels = val_data
                outputs = net(inputs.to(device))  # eval model only have last output layer
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, labels.to(device)).sum().item()
        test_acc = acc / test_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, test_acc))
        # better accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), save_path)
    print('Finished Training')


if __name__ == '__main__':
    main()
