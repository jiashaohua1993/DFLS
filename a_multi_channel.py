

import torch.nn as nn
import torch
from einops.layers.torch import Rearrange

class Multi_channel_data_preprocess(nn.Module):
    def __init__(self, in_channels=3, inner_channels=256, times=4):
        """
        Downsample the input feature map {times} times
        """
        super().__init__()
        self.initial_downsample = nn.Sequential(
            Rearrange("b1 b2 c (h p1) (w p2) -> (b1 b2) (p1 p2 c) h w", p1=times, p2=times)  # times is 4
        )

        self.conv_group1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(12, 48, 3, 1, 1),

                nn.LeakyReLU(inplace=True),# First layer: from 12 to 48
                nn.Conv2d(48, 64, 3, 1, 1),
                GlobalNorm(64),
                nn.LeakyReLU(inplace=True)# Second layer: from 48 to 64
            ) for _ in range(4)  # Create 4 such sequences
        ])

        self.conv_group2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(24, 64, 3, 1, 1),  # First layer: from 12 to 48

                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 1, 1),  # Second layer: from 48 to 64
                GlobalNorm(128),
                nn.LeakyReLU(inplace=True)
            ) for _ in range(2)  # Create 4 such sequences
        ])
        self.relu = nn.ReLU(inplace=True)

        #self.conv_group3 = nn.Conv2d(in_channels * times * times, inner_channels, 3, 1, 1)  # For the third group

        self.conv_group3 = nn.Sequential(
            nn.Conv2d(in_channels * times * times, inner_channels, 3, 1, 1),
            nn.BatchNorm2d(inner_channels),  # 添加 Batch Normalization
            nn.ReLU(inplace=True)  # 在最后加上 ReLU
        )

    def forward(self, x):
        # Initial downsample
        x = self.initial_downsample(x)  # Shape: [25, 256, 64, 64]

        # Group 1: Split into 4 slices of 12 channels each
        group1_slices = x.split(12, dim=1)  # Shape: [b, 12, 64, 64] * 4
        group1_output = [conv(slice) for conv, slice in zip(self.conv_group1, group1_slices)]

        group1_output = torch.cat(group1_output, dim=1)  # Shape: [25, 256, 64, 64]

        # Group 2: Split into 2 slices of 24 channels each
        group2_slices = x.split(24, dim=1)  # Shape: [b, 24, 64, 64] * 2
        group2_output = [conv(slice) for conv, slice in zip(self.conv_group2, group2_slices)]

        group2_output = torch.cat(group2_output, dim=1)  # Shape: [25, 256, 64, 64]
        del group1_slices
        del group2_slices

        # del group1_slices
        # del group2_slices
        # Group 3: Direct convolution
        group3_output = self.conv_group3(x)  # Shape: [25, 256, 64, 64]
        final_output = (group1_output + group2_output + group3_output) / 3
        final_output=self.relu(final_output)
        return final_output





class GlobalNorm(nn.Module):
    def __init__(self, num_features):
        super(GlobalNorm, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        # 计算均值和方差
        mean = x.mean(dim=[0, 2, 3], keepdim=True)  # 在通道和空间维度上计算均值
        var = x.var(dim=[0, 2, 3], keepdim=True)   # 在通道和空间维度上计算方差
        # 归一化
        x = (x - mean) / torch.sqrt(var + 1e-5)  # 加上小常数以防止除以0
        return x