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
parser.add_argument("-b", "--batch_size", type=int, default=16, help="学習時のバッチサイズ")
args = parser.parse_args()

# オーグメント設定
normalize = transforms.Normalize(  # データの正規化（各チャネルの平均，各チャネルの標準偏差）
    mean=[0.49139765, 0.48215759, 0.44653141],
    std=[0.24703199, 0.24348481, 0.26158789]
)
transform_test = transforms.Compose([
    transforms.ToTensor(),  # Tensor
    normalize
])

# テストデータをダウンロード
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

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
