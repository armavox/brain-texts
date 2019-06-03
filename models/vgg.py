import torch
import torch.nn as nn
import torchvision


class VGG(nn.Module):
    def __init__(self, combine_dim, from_pretrained=False):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,
                         padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,
                         padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,
                         padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,
                         padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,
                         padding=0, dilation=1, ceil_mode=False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=7)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=combine_dim, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG11(nn.Module):
    def __init__(self, combine_dim, from_pretrained=False):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1,
                         ceil_mode=False),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1,
                         ceil_mode=False),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1,
                         ceil_mode=False),
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1,
                         ceil_mode=False),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1,
                         ceil_mode=False)
        )

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=3)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=13824, out_features=4096, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=combine_dim, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        # print('INSIDE', x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print('INSIDE', x.shape)
        x = self.classifier(x)
        # print('INSIDE', x.shape)
        return x
