import os
import csv
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from pytorch_models import networks

# パーサーの設定
parser = argparse.ArgumentParser()
parser.add_argument("net", type=str, help="ネットワークモデルの名前")
parser.add_argument("-w", "--weight_path", type=str, help="学習済み重みのファイルパス")
parser.add_argument("-b", "--batch_size", type=int, default=89, help="学習時のバッチサイズ")
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
    mean = [0.1307]
    std = [0.3081]

transform_test = transforms.Compose([
    transforms.ToTensor(),  # Tensor
    transforms.Normalize(mean, std),
])

# テストデータをダウンロード
testset = torchvision.datasets.EMNIST(root='./data', split='bymerge', train=False, download=True,
                                      transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} で評価するっぴ！")

# モデルの選択
net = networks.get_net(args.net)
print(net)

# 保存した重みの読み込み
weight_path = args.weight_path
net.load_state_dict(torch.load(weight_path))

# csv の作成
print("csv ファイルを作るっぴ！")
csv_path = f"./emnist_csv/{os.path.splitext(os.path.basename(args.weight_path))[0]}.csv"
with open(csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "prediction"])  # ヘッダーの追加

image_id = 10000  # 0 ~ 9999 までは CIFAR100

# 推論結果の追記
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        # print(outputs)
        _, predicted = torch.max(outputs, 1)
        # print(predicted)

        with open(csv_path, 'a') as f:
            writer = csv.writer(f)

            for i in range(args.batch_size):
                writer.writerow([image_id, predicted[i].item()])
                image_id += 1

print("csv ファイルができたっぴ！")
