import torch.nn as nn
import torch


class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes=64, ratio=8):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * residual


class SpatialAttention3D(nn.Module):
    def __init__(self, out_channel=64, kernel_size=3, stride=1, padding=1):
        super(SpatialAttention3D, self).__init__()

        self.conv = nn.Conv3d(2, out_channel,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        out = x * residual
        return out


class residual_block(nn.Module):
    def __init__(self, channel_size=64):
        super(residual_block, self).__init__()

        self.conv = nn.Conv3d(channel_size, channel_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(channel_size)

    def forward(self, x):
        residual = x
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        out = y + residual
        return out
