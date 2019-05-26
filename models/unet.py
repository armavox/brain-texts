import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_utils3d import *


class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 1)
        self.fc1 = dense(1024, 1024)
        # self.fc2 = dense(2048, 1024)
        self.regressor = nn.Linear(1024, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.regressor(x)
        return x

# class LSTMBrain(nn.Module):
#     def __init__(self, emb_size, cat_size):
#         super().__init__()
#         self.fc1 = nn.Linear(emb_size, 1024)
#         self.fc2 = nn.Linear(1024, )
