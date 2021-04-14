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
    mean=[0.5, 0.5, 0.5],
    std=[0.25, 0.25, 0.25],
)
transform_test = transforms.Compose([
    transforms.ToTensor(),  # Tensor
    normalize
])

# テストデータをダウンロード
testset = torchvision.datasets.EMNIST(root='./data', split='bymerge', train=False, download=True,
                                      transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# EMNIST ByMergeのクラス
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z',
           'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r',
           't')

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} で評価するっぴ！")

# モデルの選択
net = networks.get_net(args.net)
print(net)

# 保存した重みの読み込み
weight_path = args.weight_path
net.load_state_dict(torch.load(weight_path))

# 評価
print("評価を始めるっぴ！")
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i in range(args.batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(len(classes)):
    print("Accuracy of %1s: %f" % (classes[i], class_correct[i] / class_total[i]))

print(f"Total Accuracy   : {sum(class_correct) / sum(class_total)}")
print("評価が終わったっぴ！")
