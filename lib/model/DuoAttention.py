import numpy as np
import torch
from torch import nn
from lib.model.attention_block import SpatialAttention3D, ChannelAttention3D, residual_block, SELayer


class DAM(nn.Module):
    def __init__(self, channels=64):
        super(DAM, self).__init__()

        self.sa = SpatialAttention3D(out_channel=channels)
        self.ca = ChannelAttention3D(in_planes=channels)
        self.se = SELayer(in_channels=channels)

    def forward(self, x):
        residual = x
        out = self.ca(x)
        out = self.sa(out)
        # out = self.se(x)
        out = out + residual
        return out


class Duo_Attention(nn.Module):
    def __init__(
            self, input_size=(1, 169, 208, 179), num_classes=3, dropout=0
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(input_size[0], 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            # nn.MaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            SELayer(in_channels=16),
            # residual_block(channel_size=16),
            # nn.MaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            SELayer(in_channels=32),
            # residual_block(channel_size=32),
            DAM(channels=32),
            # nn.MaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            SELayer(64),
            # residual_block(channel_size=64),
            # nn.MaxPool3d(2, 2),
            DAM(channels=64),

            nn.AvgPool3d(3, stride=1),
        )

        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = self.conv(input_tensor)
        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(p=dropout),
            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1024),
            nn.Linear(1024, num_classes),
            # nn.Softmax(dim=1),
        )

    def forward(self, x):
        y = torch.unsqueeze(x, dim=1)
        y = self.conv(y)
        y = self.fc(y)
        return y
