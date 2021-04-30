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
parser.add_argument("-e", "--epochs", type=int, default=1, help="学習エポック数")
parser.add_argument("-p", "--point", type=int, default=5, help="point")
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


def Generate_img(epoch, point, G_model, device, z_dim, np_labels, log_dir='data_cGAN_f'):
    G_model.eval()

    with torch.no_grad():
        # 生成に必要な乱数
        noise = torch.normal(mean=0.5, std=0.2, size=(100, z_dim)).to(device)

        # Generatorでサンプル生成
        labels = torch.Tensor(np_labels).to(torch.int64).to(device)
        samples = torch.transpose(G_model(noise, labels).data.cpu(), 2, 3)

        # npz 形式で保存
        np_samples = samples.numpy()
        np.savez(os.path.join(log_dir, f'point_{point}_epoch_{epoch}_f.npz'), x=np_samples, y=np_labels)
        save_image(samples, os.path.join(log_dir, f'point_{point}_epoch_{epoch}_f.png'), nrow=10)


def model_run(num_epochs, device):
    # Generatorに入れるノイズの次元
    z_dim = 30

    # クラス数
    num_class = 47

    # モデル定義
    point = args.point
    G_model = Generator(z_dim, num_class).to(device)
    checkpoint = torch.load('./checkpoint_cGAN/G_model_{}'.format(point))
    G_model.load_state_dict(checkpoint['model_state_dict'])

    # Generatorを試すときに使うラベルを作る
    np_labels = np.full(100, 40, dtype=np.float32)
    print("推論を始めるっぴ！")
    for epoch in range(num_epochs):
        print("epoch: %d/%d" % (epoch + 1, num_epochs))
        Generate_img(epoch + 1, point, G_model, device, z_dim, np_labels)


# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} で学習するっぴ！")

# モデルを回す
model_run(num_epochs=args.epochs, device=device)
