import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from lib.pvtv2 import pvt_v2_b2


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# Cross-level feature integration
class CFI(nn.Module):
    def __init__(self, hchannel, channel):
        super(CFI, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv3_1 = ConvBNR(channel, channel, 3)
        self.conv3_2 = ConvBNR(channel, channel, 3)
        self.conv5_1 = ConvBNR(channel, channel, 5)
        self.conv5_2 = ConvBNR(channel, channel, 5)
        self.conv3_3 = ConvBNR(channel * 2, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1_1 = ConvBNR(channel * 2, channel, 1)
        self.relu = nn.ReLU()
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv1d2 = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        size = x1.size()[2:]
        if x1.size()[2:] != x2.size()[2:]:
            x2 = F.interpolate(x2, size, mode='bilinear', align_corners=False)
        w1 = self.avg_pool(self.conv3_1(x1))
        w1 = self.conv1d(w1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        w1 = self.sigmoid(w1)
        w2 = self.avg_pool(self.conv3_2(x2))
        w2 = self.conv1d2(w2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        w2 = self.sigmoid(w2)
        x11 = w1 * x2
        x21 = w2 * x1
        x3 = self.conv1_1(torch.cat((x11, x21), dim=1))
        x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
        x12 = x3 + x1
        x22 = x3 + x2
        x_goal = self.conv3_3(torch.cat((x12, x22), dim=1))

        out = self.relu(x_goal)

        return out
class Conv_Block(nn.Module):  # [64, 128, 320, 512]
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(128 + 320 + 512, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels * 2)

        self.conv3 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse


class EF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EF, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=2, dilation=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=4, dilation=4)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=8, dilation=8)
        )
        self.conv_cat = BasicConv2d(3 * out_channel, out_channel, 3, padding=2)
        self.conv_res = BasicConv2d(in_channel, out_channel, 3,padding=2)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.mlp_gamma = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.mlp_alpha = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x, edge):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x_cat + self.conv_res(x))
        edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(edge)
        gamma = self.mlp_gamma(1 - actv)
        beta = self.mlp_beta(actv)
        alpha = self.mlp_alpha(actv)
        out = x * (1 + alpha) + beta - gamma
        out = self.relu(out)
        return out

class DIP(nn.Module):
    def __init__(self, channel):
        super(DIP, self).__init__()
        self.conv1 = BasicConv2d(512, 64, 3, padding=1)
        self.conv2 = BasicConv2d(512, 64, 3, padding=1)
        self.conv3 = BasicConv2d(320, 64, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.conv5 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv6 = nn.Conv2d(channel, 1, 1)

    def forward(self, x2, x1, x3):
        x2 = self.conv1(x2)
        x1 = self.conv2(x1)
        x3 = self.conv3(x3)
        x1_up2 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)  # 22,22
        x21 = x1_up2 * x2 + x2
        x31 = x1_up2 * x3 + x3
        x1_2 = self.conv4(torch.cat((x21, x1_up2, x31), 1))
        x4 = self.conv5(x21 + x31 + x1_2)
        x = self.conv6(x4)

        return x
class EB(nn.Module):  # [64, 128, 320, 512, 512]
    def __init__(self):
        super(EB, self).__init__()
        self.conv4 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.block = nn.Sequential(
            ConvBNR(64 + 64, 64, 3),
            nn.Conv2d(64, 1, 1))
        self.conv1 = BasicConv2d(512, 64, 3, padding=1)
        self.conv2 = BasicConv2d(128, 64, 3, padding=1)
        self.conv3 = BasicConv2d(64, 64, 3, padding=1)

    def forward(self, x5, x2, x1):
        x5 = self.conv1(x5)
        x2 = self.conv2(x2)
        x1 = self.conv3(x1)
        size = x1.size()[2:]
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size, mode='bilinear', align_corners=False)
        out1 = torch.cat((x5, x1), dim=1)
        out2 = torch.cat((x5, x2), dim=1)
        out1 = self.conv4(out1)
        out2 = self.conv5(out2)

        out = torch.cat((out1, out2), dim=1)
        out = self.block(out)

        return out

# Laddered Converged Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.backbone = pvt_v2_b2()
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.conv_block = Conv_Block(512)

        self.dip = DIP(64)

        self.eb = EB()

        self.efm1 = EF(64,64)
        self.efm2 = EF(128,64)
        self.ps5 = EF(512,64)
        self.ps4 = EF(512,64)
        self.ps3 = EF(320,64)

        self.cfi1 = CFI(64, 64)
        self.cfi2 = CFI(64, 64)
        self.cfi3 = CFI(64, 64)
        self.cfi4 = CFI(64, 64)

        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(64, 1, 1)
        self.predictor3 = nn.Conv2d(64, 1, 1)
        self.predictor4 = nn.Conv2d(64, 1, 1)

        # self.NCD = NeighborConnectionDecoder(64)

    def forward(self, x):

        image_shape = x.size()[2:]
        # backbone[]
        pvt = self.backbone(x)  # [64, 128, 320, 512, 512] [88,88  44,44  22,22 11,11, 22,22]
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        if x4.size()[2:] != x3.size()[2:]:
            x41 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear')
        if x2.size()[2:] != x3.size()[2:]:
            x21 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear')

        x5 = self.conv_block(x41, x3, x21)


        S_g = self.dip(x5, x4, x3)  # 22,11,22
        S_g = torch.sigmoid(S_g)  # 1,22,22

        x5_tu = self.ps5(x5, S_g)
        x4_tu = self.ps4(x4, S_g)
        x3_tu = self.ps3(x3, S_g)

        # Extraction Boundary Module
        edge = self.eb(x5, x2, x1)
        edge_att = torch.sigmoid(edge)

        x2_eg = self.efm2(x2, edge_att)
        x1_eg = self.efm1(x1, edge_att)

        x45 = self.cfi4(x5_tu, x4_tu)
        x345 = self.cfi3(x3_tu, x45)
        x2345 = self.cfi2(x2_eg, x345)
        x12345 = self.cfi1(x1_eg, x2345)

        o4 = self.predictor4(x45)
        o4 = F.interpolate(o4, size=image_shape, mode='bilinear', align_corners=False)
        o3 = self.predictor3(x345)
        o3 = F.interpolate(o3, size=image_shape, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x2345)
        o2 = F.interpolate(o2, size=image_shape, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x12345)
        o1 = F.interpolate(o1, size=image_shape, mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, size=image_shape, mode='bilinear', align_corners=False)
        S_g_pred = F.interpolate(S_g, size=image_shape, mode='bilinear',align_corners=False)

        return o1, o2, o3, o4, oe, S_g_pred

if __name__ == '__main__':
    model = Net().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    p1, p2, p3, p4, p5, p6 = model(input_tensor)
    print(p1.size(), p2.size(), p3.size(), p4.size(), p5.size(), p6.size())