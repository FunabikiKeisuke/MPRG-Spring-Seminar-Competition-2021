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
parser.add_argument("--calc_statistics", type=bool, default=False, help="データセットのmean, stdを計算するかどうか")
args = parser.parse_args()


# オーグメント設定
def get_statistics():
    tmp_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    data = torch.cat([d[0] for d in torch.utils.data.DataLoader(tmp_set)])
    return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


if args.calc_statistics:
    mean, std = get_statistics()  # 各チャネルの平均，各チャネルの標準偏差
    print(f"このデータセットは mean: {mean.to('cpu').detach().numpy()}, std: {std.to('cpu').detach().numpy()} だっぴ！")
else:
    mean = [0.49139765, 0.48215759, 0.44653141]
    std = [0.24703199, 0.24348481, 0.26158789]

transform_test = transforms.Compose([
    transforms.ToTensor(),  # Tensor
    transforms.Normalize(mean, std)
])

# テストデータをダウンロード
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# CIFAR-10のクラス
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
    print("Accuracy of %5s: %f" % (classes[i], class_correct[i] / class_total[i]))

print(f"Total Accuracy   : {sum(class_correct) / sum(class_total)}")
print("評価が終わったっぴ！")
