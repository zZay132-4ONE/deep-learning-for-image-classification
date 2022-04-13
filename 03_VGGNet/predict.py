"""
@description: 利用VGGNet模型进行预测
@author: Zzay
@create: 2022/04/13 14:25
"""
import os
import json
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from model import vgg


def main():
    # 设置运算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 预处理函数
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 读取图像数据
    img_path = './img/daisy.jpg'
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)  # [N, C, H, W]
    img = torch.unsqueeze(img, dim=0)  # expand batch dimension

    # 读取类别数据文件
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 建立模型
    model = vgg(model_name="vgg16", num_classes=5).to(device)
    weights_path = './vgg16Net.pth'
    assert os.path.exists(weights_path), "file '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        with torch.no_grad():  # 不再需要持久记录计算图
            output = torch.squeeze(model(img.to(device))).cpu()
            prediction = torch.softmax(output, dim=0)
            prediction_class = torch.argmax(prediction).numpy()
        print_res = "class: {};  prob: {:.3}".format(class_indict[str(prediction_class)],
                                                     prediction[prediction_class].numpy())
        plt.title(print_res)
        for i in range(len(prediction)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      prediction[i].numpy()))
        plt.show()


if __name__ == '__main__':
    main()
