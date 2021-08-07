import torch
from torch import nn
import torch.nn.functional as F


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1_1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch1x1_2 = nn.Conv2d(in_channels, 12, kernel_size=1)
        self.branch3x3_1 = nn.Conv2d(12, 16, kernel_size=3,padding=1)

        self.branch1x1_3 = nn.Conv2d(in_channels, 12, kernel_size=1,padding=1)
        self.branch5x5 = nn.Conv2d(12, 16, kernel_size=5,padding=2)

        # self.branch_pool = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.branch1x1_3 = nn.Conv2d(in_channels, 16, kernel_size=1)

    def forward(self, x):
        # print("x:",x.shape)
        branch1 = self.branch1x1_1(x)
        # print("bran_2", branch1.shape)

        branch2 = self.branch1x1_2(x)
        branch2 = self.branch3x3_1(branch2)
        # print("bran_2", branch2.shape)

        branch3 = self.branch1x1_3(x)
        branch3 = self.branch5x5(branch3)
        # print("bran_3", branch3.shape)

        branch4 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # print("bran_4", branch4.shape)
        branch4 = self.branch1x1_3(branch4)
        # print("bran_4", branch4.shape)

        outputs = [branch1, branch2, branch3, branch4]
        # print("1231313")
        return torch.cat(outputs, dim=1)

class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch1x1_1 = nn.Conv2d(in_channels, 32, kernel_size=1)

        self.branch1x1_2 = nn.Conv2d(in_channels, 24, kernel_size=1)
        self.branch3x3_1 = nn.Conv2d(24, 32, kernel_size=3,padding=1)

        self.branch1x1_3 = nn.Conv2d(in_channels, 24, kernel_size=1,padding=1)
        self.branch5x5 = nn.Conv2d(24, 32, kernel_size=5,padding=2)

        # self.branch_pool = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.branch1x1_3 = nn.Conv2d(in_channels, 32, kernel_size=1)

    def forward(self, x):
        # print("x:",x.shape)
        branch1 = self.branch1x1_1(x)
        # print("bran_2", branch1.shape)

        branch2 = self.branch1x1_2(x)
        branch2 = self.branch3x3_1(branch2)
        # print("bran_2", branch2.shape)

        branch3 = self.branch1x1_3(x)
        branch3 = self.branch5x5(branch3)
        # print("bran_3", branch3.shape)

        branch4 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # print("bran_4", branch4.shape)
        branch4 = self.branch1x1_3(branch4)
        # print("bran_4", branch4.shape)

        outputs = [branch1, branch2, branch3, branch4]
        # print("1231313")
        return torch.cat(outputs, dim=1)

class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super(InceptionC, self).__init__()
        self.branch1x1_1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        self.branch1x1_2 = nn.Conv2d(in_channels, 48, kernel_size=1)
        self.branch3x3_1 = nn.Conv2d(48, 64, kernel_size=3,padding=1)

        self.branch1x1_3 = nn.Conv2d(in_channels, 48, kernel_size=1,padding=1)
        self.branch5x5 = nn.Conv2d(48, 64, kernel_size=5,padding=2)

        # self.branch_pool = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.branch1x1_3 = nn.Conv2d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        # print("x:",x.shape)
        branch1 = self.branch1x1_1(x)
        # print("bran_2", branch1.shape)

        branch2 = self.branch1x1_2(x)
        branch2 = self.branch3x3_1(branch2)
        # print("bran_2", branch2.shape)

        branch3 = self.branch1x1_3(x)
        branch3 = self.branch5x5(branch3)
        # print("bran_3", branch3.shape)

        branch4 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # print("bran_4", branch4.shape)
        branch4 = self.branch1x1_3(branch4)
        # print("bran_4", branch4.shape)

        outputs = [branch1, branch2, branch3, branch4]
        # print("1231313")
        return torch.cat(outputs, dim=1)

class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch1x1_1 = nn.Conv2d(in_channels, 128, kernel_size=1)

        self.branch1x1_2 = nn.Conv2d(in_channels, 96, kernel_size=1)
        self.branch3x3_1 = nn.Conv2d(96, 128, kernel_size=3,padding=1)

        self.branch1x1_3 = nn.Conv2d(in_channels, 96, kernel_size=1,padding=1)
        self.branch5x5 = nn.Conv2d(96, 128, kernel_size=5,padding=2)

        # self.branch_pool = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.branch1x1_3 = nn.Conv2d(in_channels, 128, kernel_size=1)

    def forward(self, x):
        # print("x:",x.shape)
        branch1 = self.branch1x1_1(x)
        # print("bran_2", branch1.shape)

        branch2 = self.branch1x1_2(x)
        branch2 = self.branch3x3_1(branch2)
        # print("bran_2", branch2.shape)

        branch3 = self.branch1x1_3(x)
        branch3 = self.branch5x5(branch3)
        # print("bran_3", branch3.shape)

        branch4 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # print("bran_4", branch4.shape)
        branch4 = self.branch1x1_3(branch4)
        # print("bran_4", branch4.shape)

        outputs = [branch1, branch2, branch3, branch4]
        # print("1231313")
        return torch.cat(outputs, dim=1)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), #3是卷积核大小
            nn.BatchNorm2d(out_ch),                 #归一化处理，进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet, self).__init__()
        self.incep1=InceptionA(in_channels=64)
        self.incep2 = InceptionB(in_channels=128)
        self.incep3 = InceptionC(in_channels=256)
        self.incep4 = InceptionD(in_channels=512)
        self.conv1 = DoubleConv(in_ch, 64)

        self.pool1 = nn.MaxPool2d(2)    #2代kersizre=2
        self.conv2 = DoubleConv(64, 128)

        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        print("c1:",c1.shape)
        c1=self.incep1(c1)
        print("c1:", c1.shape)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        print("c2:", c2.shape)
        c2=self.incep2(c2)
        print("c2:", c2.shape)
        # print(c2.shape)

        p2=self.pool2(c2)
        c3=self.conv3(p2)#c3: torch.Size([1, 256, 128, 128])
        # print("c3:",c3.shape)
        print("c3:", c3.shape)
        c3=self.incep3(c3)
        print("c3:", c3.shape)

        p3=self.pool3(c3)
        c4=self.conv4(p3)#c4: torch.Size([1, 512, 64, 64])
        print("c4:", c4.shape)
        c4=self.incep4(c4)
        print("c4:", c4.shape)

        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)       #dim=1表示沿通道的方向将up_6和c4 concat起来
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        #out = nn.Sigmoid()(c10)
        return c10


