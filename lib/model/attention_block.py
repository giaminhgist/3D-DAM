import torch.nn as nn
import torch


class SELayer(nn.Module):
    """
    Re-implementation of the Squeeze-and-Excitation block based on:
    "Hu et al., Squeeze-and-Excitation Networks, https://arxiv.org/abs/1709.01507".
    """

    def __init__(
            self,
            in_channels: int,
            factor: int = 2,
    ):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // factor, bias=True),
            nn.ReLU(),
            nn.Linear(in_channels // factor, in_channels, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.pool(x).view(b, c)
        y = self.fc(y).view([b, c] + [1] * (x.ndim - 2))
        result = x * y

        out = result + x

        return out


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
