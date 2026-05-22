import torch
import torch.nn as nn


class Inception_V1(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, outp_1):
        super(Inception_V1,self).__init__()
        # 分支1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out1_1, 1, 1, 0),
            nn.BatchNorm2d(out1_1),
            nn.ReLU()
        )
        # 分支2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out2_1, 1, 1, 0),
            nn.BatchNorm2d(out2_1),
            nn.ReLU(),
            nn.Conv2d(out2_1, out2_3, 3, 1, 1),
            nn.BatchNorm2d(out2_3),
            nn.ReLU(),
        )
        # 分支3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out3_1, 1, 1, 0),
            nn.BatchNorm2d(out3_1),
            nn.ReLU(),
            nn.Conv2d(out3_1, out3_5, 5, 1, 2),
            nn.BatchNorm2d(out3_5),
            nn.ReLU(),
        )
        # 分支4
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_channel, outp_1, 1, 1, 0),
            nn.BatchNorm2d(outp_1),
            nn.ReLU()
        )

    def forward(self, x):
        fx1 = self.branch1(x)
        fx2 = self.branch2(x)
        fx3 = self.branch3(x)
        fx4 = self.branch4(x)
        fx = torch.cat((fx1, fx2, fx3, fx4), dim=1)
        return fx


class GoogLeNet_V1(nn.Module):
    def __init__(self):
        super(GoogLeNet_V1, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.Inception3 = nn.Sequential(
            Inception_V1(192, 64, 96, 128, 16, 32, 32),
            Inception_V1(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, 1)
        )
        self.Inception4 = nn.Sequential(
            Inception_V1(480, 192, 96, 208, 16, 48, 64),
            Inception_V1(512, 160, 112, 224, 24, 64, 64),
            Inception_V1(512, 128, 128, 256, 24, 64, 64),
            Inception_V1(512, 112, 144, 288, 32, 64, 64),
            Inception_V1(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1)
        )
        self.Inception5 = nn.Sequential(
            Inception_V1(832, 256, 160, 320, 32, 128, 128),
            Inception_V1(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(7, 1, 0),
            nn.Dropout(0.4)
        )
        self.dense = nn.Sequential(
            nn.Linear(1024, 2),
            # nn.Softmax(dim = 1)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.Inception3(x)
        x = self.Inception4(x)
        x = self.Inception5(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    googlenet = GoogLeNet_V1()
    x = torch.randn(1, 3, 224, 224)
    y = googlenet(x)
    print(y.size())

