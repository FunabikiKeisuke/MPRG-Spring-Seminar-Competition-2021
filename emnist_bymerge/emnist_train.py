import argparse
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from pytorch_models import networks

# パーサーの設定
parser = argparse.ArgumentParser()
parser.add_argument("net", type=str, help="ネットワークモデルの名前")
parser.add_argument("-e", "--epochs", type=int, default=100, help="学習エポック数")
parser.add_argument("-b", "--batch_size", type=int, default=89, help="学習時のバッチサイズ")
parser.add_argument("-a", "--best_accuracy", type=float, default=0., help="同じモデルの過去の最高精度")
parser.add_argument("--calc_statistics", type=bool, default=False, help="データセットのmean, stdを計算するかどうか")
args = parser.parse_args()


# オーグメント設定
def get_statistics():
    tmp_set = torchvision.datasets.EMNIST(root='./data', split='bymerge', train=True, download=True,
                                          transform=transforms.ToTensor())
    data = torch.cat([d[0] for d in torch.utils.data.DataLoader(tmp_set)])
    return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


if args.calc_statistics:
    mean, std = get_statistics()  # 各チャネルの平均，各チャネルの標準偏差
    print(f"このデータセットは mean: {mean.to('cpu').detach().numpy()}, std: {std.to('cpu').detach().numpy()} だっぴ！")
else:
    mean = [0.17359632]
    std = [0.33165097]

transform_train = transforms.Compose([
    # transforms.RandomAffine(degrees=75, translate=(0.3, 0.3), scale=(0.5, 1.5), shear=30),  # 微妙
    # transforms.RandomCrop(28, padding=3),  # 変わらない
    transforms.RandomPerspective(),
    # transforms.RandomRotation(10, fill=(0,)),
    transforms.RandomAffine(degrees=[-10, 10], translate=(0.1, 0.1), scale=(0.5, 1.5)),  # Rotation の代わり
    transforms.ToTensor(),  # Tensor
    transforms.Normalize(mean, std),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),  # Tensor
    transforms.Normalize(mean, std),
])

# 学習データをダウンロード
trainset = torchvision.datasets.EMNIST(root='./data', split='bymerge', train=True, download=True,
                                       transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# テストデータをダウンロード
testset = torchvision.datasets.EMNIST(root='./data', split='bymerge', train=False, download=True,
                                      transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} で学習するっぴ！")

# モデルの選択
net = networks.get_net(args.net)
print(net)

# 損失関数
criterion = nn.CrossEntropyLoss()


# 最適化関数
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


lr = 0.005
curr_lr = lr
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

# 学習
print("学習を始めるっぴ！")
epochs = args.epochs
update_acc = 0.
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
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # loss の出力
                running_loss += loss.item()
                if i % 2000 == 1999:  # iが0からのカウントなので2000イテレーションごと
                    print("iter: %d, loss: %f, time: %ds" % (i + 1, running_loss / 2000, int(time.time() - start)))
                    running_loss = 0.0
        else:  # 評価
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for data in testloader:
                    # データの取得
                    images, labels = data[0].to(device), data[1].to(device)
                    # 順伝播，クラス判定
                    outputs = net(images)
                    val_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    # 精度算出
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    acc = correct / total
            print(f"val_acc : {acc}")
            # 学習率の更新
            print(f"val_loss: {val_loss}")
            scheduler.step(val_loss)
            # if update_acc >= acc:
            #     curr_lr = lr * pow(np.random.rand(1), 3).item()
            #     print(f"精度が向上しなかったから学習率を {curr_lr} に変えるっぴ！")
            #     update_lr(optimizer, curr_lr)
            # else:
            #     update_acc = acc
            # モデルの保存
            if acc > best_acc:
                best_acc = acc
                bast_epoch = epoch
                weight_path = f"./emnist_weight/{args.net}_bs{args.batch_size}.pth"
                torch.save(net.state_dict(), weight_path)

print("学習が終わったっぴ！")
print(f"一番精度が高い epoch は {bast_epoch} だっぴ！")
