import os
import csv
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from pytorch_models import AlexNet, DenseNet, GoogLeNet, LeNet, MobileNet, MobileNetV2, ResNet, SENet, ShuffleNet, \
    VGG, WideResNet

# パーサーの設定
parser = argparse.ArgumentParser()
parser.add_argument("net", type=str, help="ネットワークモデルの名前")
parser.add_argument("-w", "--weight_path", type=str, help="学習済み重みのファイルパス")
parser.add_argument("-b", "--batch_size", type=int, default=16, help="学習時のバッチサイズ")
args = parser.parse_args()

testtransform = transforms.Compose(
    [transforms.ToTensor(),  # Tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # データの正規化（各チャネルの平均，各チャネルの標準偏差）
)

# テストデータをダウンロード
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=testtransform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# CIFAR-10のクラス
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} で評価するっぴ！")

# モデルの選択
if args.net == "AlexNet":
    net = AlexNet.AlexNet().to(device)
elif args.net == "DenseNet121":
    net = DenseNet.DenseNet121().to(device)
elif args.net == "DenseNet161":
    net = DenseNet.DenseNet161().to(device)
elif args.net == "DenseNet169":
    net = DenseNet.DenseNet169().to(device)
elif args.net == "DenseNet201":
    net = DenseNet.DenseNet201().to(device)
elif args.net == "GoogLeNet":
    net = GoogLeNet.GoogLeNet().to(device)
elif args.net == "LeNet":
    net = LeNet.LeNet().to(device)
elif args.net == "MobileNet":
    net = MobileNet.MobileNet().to(device)
elif args.net == "MobileNetV2":
    net = MobileNetV2.MobileNetV2().to(device)
elif args.net == "ResNet18":
    net = ResNet.ResNet18().to(device)
elif args.net == "ResNet34":
    net = ResNet.ResNet34().to(device)
elif args.net == "ResNet50":
    net = ResNet.ResNet50().to(device)
elif args.net == "ResNet101":
    net = ResNet.ResNet101().to(device)
elif args.net == "ResNet152":
    net = ResNet.ResNet152().to(device)
elif args.net == "SENet18":
    net = SENet.SENet18().to(device)
elif args.net == "ShuffleNetG2":
    net = ShuffleNet.ShuffleNetG2().to(device)
elif args.net == "ShuffleNetG3":
    net = ShuffleNet.ShuffleNetG3().to(device)
elif args.net == "VGG11":
    net = VGG.VGG11().to(device)
elif args.net == "VGG13":
    net = VGG.VGG13().to(device)
elif args.net == "VGG16":
    net = VGG.VGG16().to(device)
elif args.net == "VGG19":
    net = VGG.VGG19().to(device)
elif args.net == "WideResNet":
    net = WideResNet.WideResNet().to(device)
else:
    net = f"WARNING: モデル名 {args.net} がないっぴ！"
print(net)

# 保存した重みの読み込み
weight_path = args.weight_path
net.load_state_dict(torch.load(weight_path))

# csv の作成
print("csv ファイルを作るっぴ！")
csv_path = f"./cifar10_csv/{os.path.splitext(os.path.basename(args.weight_path))[0]}.csv"
with open(csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "prediction"])  # ヘッダーの追加

image_id = 0

# 推論結果の追記
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        with open(csv_path, 'a') as f:
            writer = csv.writer(f)

            for i in range(args.batch_size):
                writer.writerow([image_id, predicted[i].item()])
                image_id += 1

print("csv ファイルができたっぴ！")
