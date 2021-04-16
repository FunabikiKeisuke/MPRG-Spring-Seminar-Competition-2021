import argparse
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from pytorch_models import networks

# パーサーの設定
parser = argparse.ArgumentParser()
parser.add_argument("net", type=str, help="ネットワークモデルの名前")
parser.add_argument("predict_class", type=str, help="評価したいクラス")
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
    mean = [0.17359632]
    std = [0.33165097]

transform_train = transforms.Compose([
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=36, shuffle=True, num_workers=4)

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
predict_class_idx = classes.index(args.predict_class)

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
train_class_total = list(0. for i in range(len(classes)))
test_class_correct = list(0. for i in range(len(classes)))
test_class_total = list(0. for i in range(len(classes)))
test_class_predict = list(0. for i in range(len(classes)))
with torch.no_grad():
    # train data
    for data in trainloader:
        labels = data[1]
        for i in range(36):
            label = labels[i]
            train_class_total[label] += 1

    # test data
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i in range(args.batch_size):
            label = labels[i]
            test_class_correct[label] += c[i].item()
            test_class_total[label] += 1
            if label == predict_class_idx:  # 対称クラスの場合
                test_class_predict[predicted[i]] += 1  # 推論結果のクラスをカウントする

# 対称クラスと全クラスのデータ数を表示
print(f"The number of train-data in class {args.predict_class}     : {int(train_class_total[predict_class_idx])}")
print(f"The number of train-data for all classes: {int(sum(train_class_total))}")
print(f"The number of test-data in class {args.predict_class}      : {int(test_class_total[predict_class_idx])}")
print(f"The number of test-data for all classes : {int(sum(test_class_total))}")


# 対称クラスの推論結果グラフを作成
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(int(height)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


x = list(range(len(classes)))
fig, ax = plt.subplots()
rect = ax.bar(x, test_class_predict)
ax.set_xticks(x)
ax.set_xticklabels(classes)
autolabel(rect)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title(f"Predict: {args.predict_class}")
plt.savefig(f"./emnist_plt/{os.path.splitext(os.path.basename(args.weight_path))[0]}_{args.predict_class}.png")

# 対称クラスの精度と全クラスの精度を表示
print("Accuracy of %1s : %f" % (
    classes[predict_class_idx], test_class_correct[predict_class_idx] / test_class_total[predict_class_idx]))
print(f"Total Accuracy: {sum(test_class_correct) / sum(test_class_total)}")

print("評価が終わったっぴ！")
