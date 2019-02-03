from math import sqrt
import torch.nn as nn


class ZenkeNet(nn.Module):
    def __init__(self, num_of_datasets=6, num_of_classes=10):
        super(ZenkeNet, self).__init__()
        self.num_of_datasets = num_of_datasets
        self.num_of_classes = num_of_classes
        self.ds_idx = 0
        self.l1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.l1_relu = nn.ReLU(inplace=True)
        self.l2 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.l2_relu = nn.ReLU(inplace=True)
        self.l3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l4 = nn.Dropout(p=0.25)

        self.l5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.l5_relu = nn.ReLU(inplace=True)
        self.l6 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.l6_relu = nn.ReLU(inplace=True)
        self.l7 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l8 = nn.Dropout(p=0.25)
        self.l9 = nn.Linear(2304, 512)
        self.l9_relu = nn.ReLU(inplace=True)
        self.l10 = nn.Dropout(p=0.5)

        self.last = nn.ModuleList([nn.Linear(512, self.num_of_classes) for _ in range(0, self.num_of_datasets)])

    def set_dataset(self, ds_idx):
        self.ds_idx = ds_idx % self.num_of_datasets

    def forward(self, x):
        out = self.l1(x)
        out = self.l1_relu(out)
        out = self.l2(out)
        out = self.l2_relu(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l5_relu(out)
        out = self.l6(out)
        out = self.l6_relu(out)
        out = self.l7(out)
        out = self.l8(out)
        out = out.view(out.size(0), -1)
        out = self.l9(out)
        out = self.l9_relu(out)
        out = self.l10(out)
        out = self.last[self.ds_idx](out)
        return out


def zenke_net(**kwargs):
    num_of_datasets = kwargs.get("num_of_datasets", 6)
    num_of_classes = kwargs.get("num_of_classes", 10)
    return ZenkeNet(num_of_datasets=num_of_datasets, num_of_classes=num_of_classes)
