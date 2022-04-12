"""
@description: 调用LeNet模型进行预测
@author: Zzay
@create: 2022/04/11 01:40
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet


def main():
    # 获取预处理函数
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 建立模型，读取参数权重文件
    net = LeNet()
    net.load_state_dict(torch.load('LeNet.pth'))

    # 读取测试图像，转换维度
    im = Image.open('img/horse.jpg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    # 利用模型做出预测，与标签进行比对
    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
