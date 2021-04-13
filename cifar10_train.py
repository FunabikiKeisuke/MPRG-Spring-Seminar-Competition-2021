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
parser.add_argument("-e", "--epochs", type=int, default=1, help="学習エポック数")
parser.add_argument("-b", "--batch_size", type=int, default=16, help="学習時のバッチサイズ")
args = parser.parse_args()

# オーグメント設定
traintransform = transforms.Compose(
    [transforms.ToTensor(),  # Tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # データの正規化（各チャネルの平均，各チャネルの標準偏差）
)
testtransform = transforms.Compose(
    [transforms.ToTensor(),  # Tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # データの正規化（各チャネルの平均，各チャネルの標準偏差）
)

# 学習データをダウンロード
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=traintransform)
# 学習データをロード
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

# テストデータをダウンロード
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=testtransform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# CIFAR-10のクラス
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} で学習するっぴ！")

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

# 損失関数
criterion = nn.CrossEntropyLoss()
# 最適化関数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 学習
print("学習を始めるっぴ！")
epochs = args.epochs
best_acc = 0.0
bast_epoch = 0
for epoch in range(1, epochs + 1):
    print("epoch: %d/%d" % (epoch, epochs))
    for phase in ['train', 'test']:
        if phase == 'train':  # 学習
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # データの取得
                inputs, labels = data[0].to(device), data[1].to(device)
                # 勾配を初期化
                optimizer.zero_grad()
                # 順伝播，逆伝播，パラメータ更新
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # loss の出力
                running_loss += loss.item()
                if i % 2000 == 1999:  # iが0からのカウントなので2000イテレーションごと
                    print("iter: %d, loss: %f" % (i + 1, running_loss / 2000))
                    running_loss = 0.0
        else:  # 評価
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    # データの取得
                    images, labels = data[0].to(device), data[1].to(device)
                    # 順伝播，クラス判定
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    # 精度算出
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    acc = correct / total
            print(f"Accuracy: {acc}")
            if acc > best_acc:
                # モデルの保存
                best_acc = acc
                bast_epoch = epoch
                weight_path = f"./cifar10_weight/{args.net}_bs{args.batch_size}.pth"
                torch.save(net.state_dict(), weight_path)

print("学習が終わったっぴ！")
print(f"一番精度が高い epoch は {bast_epoch} だっぴ！")
