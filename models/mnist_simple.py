import torch.nn.functional as func
import torch.nn as nn


class FC2Layers(nn.Module):
    def __init__(self, **kwargs):
        super(FC2Layers, self).__init__()
        layer1_width = kwargs.get("layer1_width", 400)
        layer2_width = kwargs.get("layer2_width", 400)

        self.ds_idx = 0
        self.num_of_datasets = kwargs.get("num_of_datasets", 1)
        self.num_of_classes = kwargs.get("num_of_classes", 10)
        self.input_size = kwargs.get("input_size", 784)

        self.layer1 = nn.Sequential(nn.Linear(self.input_size, layer1_width))
        self.layer2 = nn.Sequential(nn.Linear(layer1_width, layer2_width))
        self.last_layer = nn.ModuleList([nn.Linear(layer2_width, self.num_of_classes)
                                         for _ in range(0, self.num_of_datasets)])

    def set_dataset(self, ds_idx):
        self.ds_idx = ds_idx % self.num_of_datasets

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = self.layer1(x)
        out = func.relu(out)
        out = self.layer2(out)
        out = func.relu(out)
        out = self.last_layer[self.ds_idx](out)
        return out


def mnist_simple_net(**kwargs):
    return FC2Layers(**kwargs)


def mnist_simple_net_400width_classlearning_1024input_10cls_1ds(**kwargs):
    return FC2Layers(input_size=1024, layer1_width=400, layer2_width=400, num_of_datasets=1,
                     num_of_classes=10, **kwargs)


def mnist_simple_net_400width_tasklearning_1024input_2cls_5ds(**kwargs):
    return FC2Layers(input_size=1024, layer1_width=400, layer2_width=400, num_of_datasets=5,
                     num_of_classes=2, **kwargs)


def mnist_simple_net_400width_domainlearning_1024input_2cls_1ds(**kwargs):
    return FC2Layers(input_size=1024, layer1_width=400, layer2_width=400, num_of_datasets=1,
                     num_of_classes=2, **kwargs)


def mnist_simple_net_400width_tasklearning_1024input_10cls_10ds(**kwargs):
    return FC2Layers(input_size=1024, layer1_width=400, layer2_width=400, num_of_datasets=10,
                     num_of_classes=10, **kwargs)


def mnist_simple_net_1000width_tasklearning_1024input_10cls_10ds(**kwargs):
    return FC2Layers(input_size=1024, layer1_width=1000, layer2_width=1000, num_of_datasets=10,
                     num_of_classes=10, **kwargs)


def mnist_simple_net_400width_domainlearning_1024input_10cls_1ds(**kwargs):
    return FC2Layers(input_size=1024, layer1_width=400, layer2_width=400, num_of_datasets=1,
                     num_of_classes=10, **kwargs)


def mnist_simple_net_1000width_domainlearning_1024input_10cls_1ds(**kwargs):
    return FC2Layers(input_size=1024, layer1_width=1000, layer2_width=1000, num_of_datasets=1,
                     num_of_classes=10, **kwargs)


def mnist_simple_net_200width_domainlearning_1024input_10cls_1ds(**kwargs):
    return FC2Layers(input_size=1024, layer1_width=200, layer2_width=200, num_of_datasets=1,
                     num_of_classes=10, **kwargs)


def mnist_simple_net_200width_domainlearning_784input_10cls_1ds(**kwargs):
    return FC2Layers(input_size=784, layer1_width=200, layer2_width=200, num_of_datasets=1,
                     num_of_classes=10, **kwargs)


def mnist_simple_net_1000width_classlearning_1024input_100cls_1ds(**kwargs):
    return FC2Layers(input_size=1024, layer1_width=1000, layer2_width=1000, num_of_datasets=1,
                     num_of_classes=100, **kwargs)
