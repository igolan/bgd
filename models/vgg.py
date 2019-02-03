# VGG11/13/16/19-like in Pytorch.
# Based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, vgg_name, use_bn=True, **kwargs):
        super(VGG, self).__init__()
        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG16_zhang': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        self.vgg_name = vgg_name
        self.features = self._make_layers(cfg[vgg_name], use_bn=use_bn)
        if self.vgg_name == 'VGG16_zhang':
            self.lastlayer_classifier = nn.Linear(256, kwargs.get("num_classes", 10))
        else:
            self.lastlayer_classifier = nn.Linear(512, kwargs.get("num_classes", 10))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.lastlayer_classifier(out)
        return out

    @staticmethod
    def _make_layers(cfg, use_bn=True):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if use_bn:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11(**kwargs):
    return VGG('VGG11')


def vgg11_no_bn(**kwargs):
    return VGG('VGG11', use_bn=False)


def vgg16_no_bn(**kwargs):
    return VGG('VGG16', use_bn=False)


def vgg16zhang_no_bn(**kwargs):
    return VGG('VGG16_zhang', use_bn=False)


def vgg16zhang_with_bn(**kwargs):
    return VGG('VGG16_zhang', use_bn=True)
