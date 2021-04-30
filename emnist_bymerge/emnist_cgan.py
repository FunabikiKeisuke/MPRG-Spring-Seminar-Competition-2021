import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
from pytorch_models import networks

# パーサーの設定
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=100, help="学習エポック数")
parser.add_argument("-b", "--batch_size", type=int, default=256, help="学習時のバッチサイズ")
args = parser.parse_args()


class Generator(nn.Module):
    def __init__(self, z_dim, num_class):
        super(Generator, self).__init__()
        self.num_class = num_class

        self.fc1 = nn.Linear(z_dim, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.LReLU1 = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(num_class, 1500)
        self.bn2 = nn.BatchNorm1d(1500)
        self.LReLU2 = nn.LeakyReLU(0.2)

        self.fc3 = nn.Linear(1800, 128 * 7 * 7)
        self.bn3 = nn.BatchNorm1d(128 * 7 * 7)
        self.bo1 = nn.Dropout(p=0.5)
        self.LReLU3 = nn.LeakyReLU(0.2)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # チャネル数を128⇒64に変える。
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # チャネル数を64⇒1に変更
            nn.Tanh(),
        )

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()

    def forward(self, noise, labels):
        y_1 = self.fc1(noise)
        y_1 = self.bn1(y_1)
        y_1 = self.LReLU1(y_1)

        one_hot_label = nn.functional.one_hot(labels, num_classes=self.num_class).to(torch.float32)  # [256, 47]
        y_2 = self.fc2(one_hot_label)
        y_2 = self.bn2(y_2)
        y_2 = self.LReLU2(y_2)

        x = torch.cat([y_1, y_2], 1)
        x = self.fc3(x)
        x = self.bo1(x)
        x = self.LReLU3(x)
        x = x.view(-1, 128, 7, 7)
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, num_class):
        super(Discriminator, self).__init__()
        self.num_class = num_class

        self.conv = nn.Sequential(
            nn.Conv2d(num_class + 1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()

    def forward(self, img, labels):
        one_hot_label = nn.functional.one_hot(labels, num_classes=self.num_class).to(torch.float32)  # [256, 47]
        y_2 = one_hot_label.view(-1, self.num_class, 1, 1)  # [256, 47, 1, 1]
        y_2 = y_2.expand(-1, -1, 28, 28)  # [256, 47, 28, 28]

        x = torch.cat([img, y_2], 1)

        x = self.conv(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)
        return x


def train_func(D_model, G_model, batch_size, z_dim, num_class, criterion, D_optimizer, G_optimizer, data_loader,
               device):
    # 訓練モード
    D_model.train()
    G_model.train()

    # 本物のラベルは1
    y_real = torch.ones((batch_size, 1)).to(device)
    D_y_real = (torch.rand((batch_size, 1)) / 2 + 0.7).to(device)  # Dに入れるノイズラベル

    # 偽物のラベルは0
    y_fake = torch.zeros((batch_size, 1)).to(device)
    D_y_fake = (torch.rand((batch_size, 1)) * 0.3).to(device)  # Dに入れるノイズラベル

    # lossの初期化
    D_running_loss = 0
    G_running_loss = 0

    # バッチごとの計算
    for batch_idx, (data, labels) in enumerate(data_loader):
        # バッチサイズに満たない場合は無視
        if data.size()[0] != batch_size:
            break

        # ノイズ作成
        z = torch.normal(mean=0.5, std=0.2, size=(batch_size, z_dim))  # 平均0.5の正規分布に従った乱数を生成

        real_img, label, z = data.to(device), labels.to(device), z.to(device)

        # Discriminatorの更新
        D_optimizer.zero_grad()

        # Discriminatorに本物画像を入れて順伝播⇒Loss計算
        D_real = D_model(real_img, label)
        D_real_loss = criterion(D_real, D_y_real)

        # DiscriminatorにGeneratorにノイズを入れて作った画像を入れて順伝播⇒Loss計算
        fake_img = G_model(z, label)
        D_fake = D_model(fake_img.detach(), label)  # fake_imagesで計算したLossをGeneratorに逆伝播させないように止める
        D_fake_loss = criterion(D_fake, D_y_fake)

        # 2つのLossの和を最小化
        D_loss = D_real_loss + D_fake_loss

        D_loss.backward()
        D_optimizer.step()

        D_running_loss += D_loss.item()

        # Generatorの更新
        G_optimizer.zero_grad()

        # Generatorにノイズを入れて作った画像をDiscriminatorに入れて順伝播⇒見破られた分がLossになる
        fake_img_2 = G_model(z, label)
        D_fake_2 = D_model(fake_img_2, label)

        # Gのloss(max(log D)で最適化)
        G_loss = -criterion(D_fake_2, y_fake)

        G_loss.backward()
        G_optimizer.step()
        G_running_loss += G_loss.item()

    D_running_loss /= len(data_loader)
    G_running_loss /= len(data_loader)

    return D_running_loss, G_running_loss


def Generate_img(epoch, G_model, device, z_dim, noise, var_mode, labels, log_dir='logs_cGAN'):
    G_model.eval()

    with torch.no_grad():
        if var_mode == True:
            # 生成に必要な乱数
            noise = torch.normal(mean=0.5, std=0.2, size=(47, z_dim)).to(device)
        else:
            noise = noise

        # Generatorでサンプル生成
        samples = torch.transpose(G_model(noise, labels).data.cpu(), 2, 3)
        # samples = (samples / 2) + 0.5
        save_image(samples, os.path.join(log_dir, 'epoch_%05d.png' % (epoch + 1)), nrow=7)


def model_run(num_epochs, batch_size, dataloader, device):
    # Generatorに入れるノイズの次元
    z_dim = 30
    var_mode = False  # 表示結果を見るときに毎回異なる乱数を使うかどうか
    # 生成に必要な乱数
    noise = torch.normal(mean=0.5, std=0.2, size=(47, z_dim)).to(device)

    # クラス数
    num_class = 47

    # Generatorを試すときに使うラベルを作る
    labels = np.arange(num_class, dtype=np.float32)
    label = torch.Tensor(labels).to(torch.int64).to(device)

    # モデル定義
    D_model = Discriminator(num_class).to(device)
    G_model = Generator(z_dim, num_class).to(device)

    # lossの定義(引数はtrain_funcの中で指定)
    criterion = nn.BCELoss().to(device)

    # optimizerの定義
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, weight_decay=1e-5,
                                   amsgrad=False)
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, weight_decay=1e-5,
                                   amsgrad=False)

    D_loss_list = []
    G_loss_list = []

    print("学習を始めるっぴ！")
    all_time = time.time()
    for epoch in range(num_epochs):
        print("epoch: %d/%d" % (epoch + 1, num_epochs))
        start_time = time.time()

        D_loss, G_loss = train_func(D_model, G_model, batch_size, z_dim, num_class, criterion, D_optimizer, G_optimizer,
                                    dataloader, device)

        D_loss_list.append(D_loss)
        G_loss_list.append(G_loss)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        # エポックごとに結果を表示
        print("D_loss: %4f, G_loss: %4f, time: %d min %d sec" % (D_loss, G_loss, mins, secs))

        if (epoch + 1) % 1 == 0:
            Generate_img(epoch, G_model, device, z_dim, noise, var_mode, label)

        # モデル保存のためのcheckpointファイルを作成
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': G_model.state_dict(),
                'optimizer_state_dict': G_optimizer.state_dict(),
                'loss': G_loss,
            }, './checkpoint_cGAN/G_model_{}'.format(epoch + 1))

    return D_loss_list, G_loss_list


# 学習データをダウンロード
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.EMNIST(root='./data', split='bymerge', train=True, download=True, transform=transform)

# データローダー
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} で学習するっぴ！")

# モデルを回す
D_loss_list, G_loss_list = model_run(num_epochs=args.epochs, batch_size=args.batch_size, dataloader=trainloader,
                                     device=device)
