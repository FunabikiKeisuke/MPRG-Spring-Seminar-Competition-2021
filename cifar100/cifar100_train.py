import argparse
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from pytorch_models import networks
from utility.log import Log
from utility.initialize import initialize
from utility.cutout import Cutout
from utility.sam import SAM
from utility.step_lr import StepLR
from utility.smooth_cross_entropy import smooth_crossentropy

# パーサーの設定
parser = argparse.ArgumentParser()
parser.add_argument("net", type=str, help="ネットワークモデルの名前")
parser.add_argument("-e", "--epochs", type=int, default=200, help="学習エポック数")
parser.add_argument("-b", "--batch_size", type=int, default=100, help="学習時のバッチサイズ")
parser.add_argument("-a", "--best_accuracy", type=float, default=0., help="同じモデルの過去の最高精度")
parser.add_argument("--calc_statistics", type=bool, default=False, help="データセットのmean, stdを計算するかどうか")
args = parser.parse_args()

# 初期化
initialize(args, seed=42)


# オーグメント設定
def get_statistics():
    tmp_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
    data = torch.cat([d[0] for d in torch.utils.data.DataLoader(tmp_set)])
    return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


if args.calc_statistics:
    mean, std = get_statistics()  # 各チャネルの平均，各チャネルの標準偏差
    print(f"このデータセットは mean: {mean.to('cpu').detach().numpy()}, std: {std.to('cpu').detach().numpy()} だっぴ！")
else:
    mean = [0.5070758, 0.4865503, 0.44091913]
    std = [0.26733428, 0.25643846, 0.27615047]

transform_train = transforms.Compose([
    # transforms.RandomRotation(15),
    transforms.RandomCrop(size=(32, 32), padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Tensor
    transforms.Normalize(mean, std),
    Cutout()
])
transform_test = transforms.Compose([
    transforms.ToTensor(),  # Tensor
    transforms.Normalize(mean, std),
])

# 学習データをダウンロード
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# テストデータをダウンロード
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} で学習するっぴ！")

# モデルの選択
net = networks.get_net(args.net)
print(net)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


print(count_parameters(net))
# 損失関数
# criterion = nn.CrossEntropyLoss()
# 最適化関数
base_optimizer = optim.SGD
optimizer = SAM(net.parameters(), base_optimizer, rho=0.05, lr=0.1, momentum=0.9, weight_decay=0.0005)
scheduler = StepLR(optimizer, 0.1, args.epochs)

# 学習
print("学習を始めるっぴ！")
epochs = args.epochs
best_acc = args.best_accuracy
bast_epoch = 0
start = time.time()
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
                loss = smooth_crossentropy(outputs, labels)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)
                smooth_crossentropy(net(inputs), labels).mean().backward()
                optimizer.second_step(zero_grad=True)
                # 学習率更新
                with torch.no_grad():
                    # correct = torch.argmax(outputs.data, 1) == labels
                    scheduler(epoch)
                # loss の出力
                running_loss += loss.mean().item()
                if i % 200 == 199:  # iが0からのカウントなので200イテレーションごと
                    print("iter: %d, loss: %f, time: %ds" % (i + 1, running_loss / 2000, int(time.time() - start)))
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
                weight_path = f"./cifar100_weight/{args.net}_bs{args.batch_size}.pth"
                torch.save(net.state_dict(), weight_path)

print("学習が終わったっぴ！")
print(f"一番精度が高い epoch は {bast_epoch} だっぴ！")
