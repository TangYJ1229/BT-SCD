import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3, groups=in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )
    def forward(self, x):
        return self.dwconv(x)


class CBA1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.cba(x)


class CBA3x3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.cba(x)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ECA(nn.Module):
    def __init__(self, kernal=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernal, padding=(kernal - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class task_interaction_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sem2Change = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction='mean')

    def forward(self, old_sem1, old_sem2, old_change, change_result):
        sem_max_out, _ = torch.max(torch.abs(old_sem1 - old_sem2), dim=1, keepdim=True)
        sem_avg_out = torch.mean(torch.abs(old_sem1 - old_sem2), dim=1, keepdim=True)
        sem_out = self.sigmoid(self.Sem2Change(torch.cat([sem_max_out, sem_avg_out], dim=1)))
        new_change = old_change * sem_out

        b, c, h, w = old_sem1.size()
        fea_sem1 = torch.reshape(old_sem1.permute(0,2,3,1), [b*h*w, c])
        fea_sem2 = torch.reshape(old_sem2.permute(0,2,3,1), [b*h*w, c])

        change_mask = torch.argmax(change_result, dim=1)
        unchange_mask = ~change_mask.bool()
        target = unchange_mask.float()
        target = target - change_mask.float()
        target = torch.reshape(target, [b * h * w])
        similarity_loss = self.loss_f(fea_sem1, fea_sem2, target)
        return new_change, similarity_loss


class decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        #upsample
        self.upconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.catconv = CBA3x3(out_channel * 2, out_channel)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, up, skip):
        #upsample
        up = self.upconv(up)
        up = torch.cat([up, skip], dim=1)
        up = self.catconv(up)
        return up

# BCFE
class Change_Specific_Transfer(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv = CBA1x1(in_channel*2, in_channel)
        self.eca = ECA()
        self.resblock = self._make_layer(ResBlock, 256, 128, 6, stride=1)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        xc1 = self.conv(torch.cat([x1, x2], dim=1))
        xc2 = self.conv(torch.cat([x2, x1], dim=1))
        change = self.eca(xc1 + xc2)
        diff = torch.abs(x1 - x2)
        change = torch.cat([change, diff], dim=1)
        change = self.resblock(change)
        return change


class Multi_Level_Feature_Aggreagation(nn.Module):

    def __init__(self,):
        super(Multi_Level_Feature_Aggreagation, self).__init__()

        self.proj1 = DWConv(512, 128)
        self.proj2 = DWConv(256, 128)

        self.cat_conv = CBA1x1(384, 128)


    def forward(self, x1, x2, x3):
        x3 = self.proj1(x3)
        x2 = self.proj2(x2)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.cat_conv(x)
        return x


class Boundary_Decoder(nn.Module):
    def __init__(self):
        super(Boundary_Decoder, self).__init__()
        self.sobel_x, self.sobel_y = get_sobel(64, 1)
        self.conv = nn.Conv2d(64, 64, 1, 1, 0)

    def forward(self, x, size):
        x = F.upsample(x, size, mode='bilinear')
        x = run_sobel(self.sobel_x, self.sobel_y, x)
        x = self.conv(x)
        return x


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y