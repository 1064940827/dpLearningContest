import torch
from torch import nn
import torch.nn.functional as F


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super(Downsample, self).__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.layers(x)


class Upsample(nn.Module):
    def __init__(self, channel):
        super(Upsample, self).__init__()
        self.layers = nn.Conv2d(channel, channel // 2, 1)

    def forward(self, x, feature_map):
        upsampled = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layers(upsampled)
        return torch.cat((out, feature_map), 1)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(1, 64)
        self.d1 = Downsample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = Downsample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = Downsample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = Downsample(512)
        self.c5 = Conv_Block(512, 1024)
        self.u1 = Upsample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = Upsample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = Upsample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = Upsample(128)
        self.c9 = Conv_Block(128, 64)
        self.out = nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, x):
        r1 = self.c1(x)
        r2 = self.c2(self.d1(r1))
        r3 = self.c3(self.d2(r2))
        r4 = self.c4(self.d3(r3))
        r5 = self.c5(self.d4(r4))
        o1 = self.c6(self.u1(r5, r4))
        o2 = self.c7(self.u2(o1, r3))
        o3 = self.c8(self.u3(o2, r2))
        o4 = self.c9(self.u4(o3, r1))
        return self.out(o4)


if __name__ == '__main__':
    x = torch.rand(1, 1, 192, 192)
    net = UNet(4)
    print(net(x).shape)
