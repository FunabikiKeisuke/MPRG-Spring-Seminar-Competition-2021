import torch
from pytorch_models import AlexNet, DenseNet, GoogLeNet, LeNet, MobileNet, MobileNetV2, ResNet, SENet, ShuffleNet, \
    VGG, WideResNet, EfficientNet

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_net(net_name):
    if net_name == "AlexNet":
        net = AlexNet.AlexNet().to(device)
    # elif net_name == "DenseNet121":
    #     net = DenseNet.DenseNet121().to(device)
    # elif net_name == "DenseNet161":
    #     net = DenseNet.DenseNet161().to(device)
    # elif net_name == "DenseNet169":
    #     net = DenseNet.DenseNet169().to(device)
    # elif net_name == "DenseNet201":
    #     net = DenseNet.DenseNet201().to(device)
    # elif net_name == "GoogLeNet":
    #     net = GoogLeNet.GoogLeNet().to(device)
    # elif net_name == "LeNet":
    #     net = LeNet.LeNet().to(device)
    # elif net_name == "MobileNet":
    #     net = MobileNet.MobileNet().to(device)
    # elif net_name == "MobileNetV2":
    #     net = MobileNetV2.MobileNetV2().to(device)
    elif net_name == "ResNet18":
        net = ResNet.ResNet18().to(device)
    elif net_name == "ResNet34":
        net = ResNet.ResNet34().to(device)
    elif net_name == "ResNet50":
        net = ResNet.ResNet50().to(device)
    elif net_name == "ResNet101":
        net = ResNet.ResNet101().to(device)
    elif net_name == "ResNet152":
        net = ResNet.ResNet152().to(device)
    # elif net_name == "SENet18":
    #     net = SENet.SENet18().to(device)
    # elif net_name == "ShuffleNetG2":
    #     net = ShuffleNet.ShuffleNetG2().to(device)
    # elif net_name == "ShuffleNetG3":
    #     net = ShuffleNet.ShuffleNetG3().to(device)
    # elif net_name == "VGG11":
    #     net = VGG.VGG11().to(device)
    # elif net_name == "VGG13":
    #     net = VGG.VGG13().to(device)
    # elif net_name == "VGG16":
    #     net = VGG.VGG16().to(device)
    # elif net_name == "VGG19":
    #     net = VGG.VGG19().to(device)
    elif net_name == "WideResNet16":
        net = WideResNet.WideResNet16().to(device)
    elif net_name == "WideResNet28":
        net = WideResNet.WideResNet28().to(device)
    elif net_name == "WideResNet40":
        net = WideResNet.WideResNet40().to(device)
    elif net_name == "EfficientNet_b0":
        net = EfficientNet.efficientnet_b0(100).to(device)
    elif net_name == "EfficientNet_b1":
        net = EfficientNet.efficientnet_b1(100).to(device)
    elif net_name == "EfficientNet_b2":
        net = EfficientNet.efficientnet_b2(100).to(device)
    elif net_name == "EfficientNet_b3":
        net = EfficientNet.efficientnet_b3(100).to(device)
    elif net_name == "EfficientNet_b4":
        net = EfficientNet.efficientnet_b4(100).to(device)
    elif net_name == "EfficientNet_b5":
        net = EfficientNet.efficientnet_b5(100).to(device)
    elif net_name == "EfficientNet_b6":
        net = EfficientNet.efficientnet_b6(100).to(device)
    elif net_name == "EfficientNet_b7":
        net = EfficientNet.efficientnet_b7(100).to(device)
    else:
        net = f"WARNING: モデル名 {net_name} がないっぴ！"

    return net
