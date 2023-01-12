#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:36:52 2022

@author: Yangzhuobin

E-mail: yzb_98@tju.edu.cn

"""

import torch
import torch.nn as nn
import numpy as np

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class cbam_block(nn.Module):

    def __init__(self, channel, ratio=4, kernel_size=3):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

def channel_shuffle(x, groups):

    # input shape: [batch_size, channels, H, W]
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x

class ChannelShuffle(nn.Module):

    def __init__(self, channels, groups):
        super(ChannelShuffle, self).__init__()
        if channels % groups != 0:
            raise ValueError("The number of channels must be divisible by the number of groups.")
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)

def Computing_mean(x, mask):

    mask = torch.count_nonzero(mask, dim=2)
    mask = torch.unsqueeze(mask, dim=2)
    x = x.sum(dim=2, keepdim=True)
    x = x / mask
    return x

class CNN(nn.Module):

    def __init__(self, F1: int, C: int, T: int, classes_num: int, D: int = 2):

        super(CNN, self).__init__()
        self.drop_out = 0.25
        self.att = cbam_block(D * F1)
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((7, 7, 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 16),
                stride=(1, 2),
                bias=False
            ),
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 8))
        )
        self.block_2 = nn.Sequential(
            nn.ZeroPad2d((7, 7, 0, 0)),
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1,
                kernel_size=(1, 16),
                stride=(1, 2),
                bias=False,
                groups=F1
            ),
            nn.Conv2d(
                in_channels=F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(D * F1),
            nn.ReLU(inplace=True)
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * F1,
                kernel_size=(3, 1),
                stride=(1, 1),
                groups=D * F1,
                bias=False
            ),
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=4,
                bias=False
            ),
            nn.BatchNorm2d(D * D * F1),
            nn.ReLU(inplace=True),
            ChannelShuffle(D * D * F1, 4),
        )
        self.block_4 = nn.Sequential(
            nn.ZeroPad2d((4, 3, 0, 0)),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, 8),
                stride=(1, 1),
                groups=D * D * F1,
                bias=False
            ),
            nn.BatchNorm2d(D * D * F1),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=4,
                bias=False
            ),
            nn.BatchNorm2d(D * D * D * F1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 16))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(D * D * D * F1, classes_num),
        )

    def forward(self, x):

        mask = torch.abs(x).sum(dim=3, keepdim=True)
        mask = (mask > 0).type(torch.float)
        x = self.block_1(x)
        x = self.block_2(x)
        x = x * mask
        x1 = Computing_mean(x, mask)
        x2 = torch.norm(x, p=2, dim=2, keepdim=True)
        x3 = torch.norm(x, p=np.inf, dim=2, keepdim=True)
        x = torch.cat([x1, x2, x3], 2)
        x = self.att(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = x.view(x.shape[0], -1)

        x = self.classifier(x)

        return x
