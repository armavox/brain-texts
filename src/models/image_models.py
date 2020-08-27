import torch.nn as nn
import torchvision as tv
import pretrainedmodels


class VGG19(nn.Module):
    def __init__(self, in_ch=1, out_cl=2, pretrained=False):
        super().__init__()
        model = tv.models.vgg19_bn()
        self.features = model.features
        self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.features[3] = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.features[7] = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

        self.avgpool = model.avgpool

        self.classifier = model.classifier
        self.classifier[-1] = nn.Linear(in_features=4096, out_features=out_cl, bias=True)

    def forward(self, x, return_feats=False):
        features = self.features(x)
        features = self.avgpool(features)
        out = self.classifier(features.view(features.size(0), -1))
        if return_feats:
            return out, features
        else:
            return out


class ResNet18(nn.Module):
    def __init__(self, in_ch=1, out_cl=2, pretrained=False):
        super().__init__()
        self.m = tv.models.resnet18()
        self.m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.m.fc = nn.Linear(in_features=512, out_features=out_cl, bias=True)

    def forward(self, x):
        return self.m(x)


class ResNet101(nn.Module):
    def __init__(self, in_ch=1, out_cl=2, pretrained=False):
        super().__init__()
        self.m = tv.models.resnet101()
        self.m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.m.fc = nn.Linear(in_features=2048, out_features=out_cl, bias=True)

    def forward(self, x):
        return self.m(x)


class ResNet152(nn.Module):
    def __init__(self, in_ch=1, out_cl=2, pretrained=False):
        super().__init__()
        self.m = tv.models.resnet152()
        self.m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.m.fc = nn.Linear(in_features=2048, out_features=out_cl, bias=True)

    def forward(self, x):
        return self.m(x)


class ResNeXt(nn.Module):
    def __init__(self, in_ch=1, out_cl=2, pretrained=False):
        super().__init__()
        self.m = tv.models.resnext101_32x8d()
        self.m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.m.fc = nn.Linear(in_features=2048, out_features=out_cl, bias=True)

    def forward(self, x):
        return self.m(x)


class Densenet121(nn.Module):
    def __init__(self, in_ch=1, out_cl=2, pretrained=False):
        super().__init__()
        self.m = tv.models.densenet121(pretrained)
        self.m.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.m.classifier = nn.Linear(in_features=1024, out_features=out_cl, bias=True)

    def forward(self, x):
        return self.m(x)


class Densenet201(nn.Module):
    def __init__(self, in_ch=1, out_cl=2, pretrained=False):
        super().__init__()
        self.m = tv.models.densenet201()
        self.m.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.m.classifier = nn.Linear(in_features=1920, out_features=out_cl, bias=True)

    def forward(self, x):
        return self.m(x)


class SE_ResNeXt101_32x4d(nn.Module):
    def __init__(self, in_ch=1, out_cl=2, pretrained=None):
        super().__init__()
        self.m = pretrainedmodels.se_resnext101_32x4d(num_classes=2, pretrained=pretrained)
        self.m.layer0.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return self.m(x)
