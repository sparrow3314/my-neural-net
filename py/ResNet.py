import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride = 1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)#注意这里是直接＋的而不是cat，直接加需要特征图大小和通道数都相同
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.resblock1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64)
        )
        self.resblock2 = nn.Sequential(
            ResBlock(64, 128, 2),
            ResBlock(128, 128)
        )
        self.resblock3 = nn.Sequential(
            ResBlock(128, 256, 2),
            ResBlock(256, 256)
        )
        self.resblock4 = nn.Sequential(
            ResBlock(256, 512, 2),
            ResBlock(512, 512),
            nn.AvgPool2d(7, 1, 0)
        )
        self.dense = nn.Sequential(
            nn.Linear(512, 1000),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    resnet = ResNet18()
    x = torch.randn(1, 3, 224, 224)
    y = resnet(x)
    print(y.size())
