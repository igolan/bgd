import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, **kwargs):
        super(LeNet, self).__init__()
        self.num_of_datasets = kwargs.get("num_of_datasets", 1)
        self.num_of_classes = kwargs.get("num_of_classes", 10)
        self.ds_idx = 0

        self.l1 = nn.Conv2d(3, 20, kernel_size=5, padding=1)
        self.l1_relu = nn.ReLU(inplace=True)
        self.l1_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l2 = nn.Conv2d(20, 50, kernel_size=5, padding=1)
        self.l2_relu = nn.ReLU(inplace=True)
        self.l2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear = nn.Sequential(nn.Linear(1800, 500))

        self.last = nn.ModuleList([nn.Linear(500, self.num_of_classes) for _ in range(0, self.num_of_datasets)])

    def set_dataset(self, ds_idx):
        self.ds_idx = ds_idx % self.num_of_datasets

    def forward(self, x):
        out = self.l1(x)
        out = self.l1_relu(out)
        out = self.l1_maxpool(out)

        out = self.l2(out)
        out = self.l2_relu(out)
        out = self.l2_maxpool(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        out = self.last[self.ds_idx](out)
        return out


def lenet(**kwargs):
    return LeNet(num_of_datasets=5, **kwargs)
