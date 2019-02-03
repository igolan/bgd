import torch
import numpy as np
import random
from torch.nn.parallel.data_parallel import DataParallel
from time import time


def get_model(model):
    if isinstance(model, DataParallel):
        return model.module
    return model


def set_seed(seed, fully_deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fully_deterministic:
            torch.backends.cudnn.deterministic = True


def bn_exists_in_net(net):
    """
    Check if the network has a BatchNormalization layer
    :param net: network
    :return: True if a 2d BatchNorm layer exists
    """
    for l in get_model(net).modules():
        if type(l) == torch.nn.modules.batchnorm.BatchNorm2d:
            return True
    return False


def save_bn_running(net):
    """
    Save running mean and variance for 2d BatchNorm layers of a network
    :param net: network
    :return: List with running means and running vars
    """
    means = [l.running_mean.clone() for l in get_model(net).modules() if
             type(l) == torch.nn.modules.batchnorm.BatchNorm2d]
    variances = [l.running_var.clone() for l in get_model(net).modules() if
                 type(l) == torch.nn.modules.batchnorm.BatchNorm2d]
    return [means, variances]


def load_bn_running(net, runnings):
    """
    Restore  running mean and variance for 2d BatchNorm layers of a network
    :param net: net
    :param runnings: List generated from save_runnings() - runnings[0] is the running mean, runnings[1] is the variance
    :return:
    """
    if runnings is None:
        return

    means = [l.running_mean for l in get_model(net).modules() if type(l) == torch.nn.modules.batchnorm.BatchNorm2d]
    variances = [l.running_var for l in get_model(net).modules() if type(l) == torch.nn.modules.batchnorm.BatchNorm2d]
    for m, m_load in zip(means, runnings[0]):
        m.copy_(m_load)
    for v, v_load in zip(variances, runnings[1]):
        v.copy_(v_load)


class AverageTracker:
    def __init__(self):
        self.n = 0
        self.avg = float(0)

    def reset(self):
        self.n = 0
        self.avg = float(0)
        return self

    def add(self, val, n=1):
        if n == 0:
            return self
        self.avg = ((self.avg * self.n) / (self.n + n)) + ((float(val) * n) / (self.n + n))
        self.n += n
        return self

    def copy(self):
        cp = AverageTracker()
        cp.n = self.n
        cp.avg = self.avg
        return cp

    def __iadd__(self, other):
        assert(type(other) == AverageTracker)
        self.add(other.avg, other.n)
        return self


class TimeRecorder:
    def __init__(self, sync_cuda=False):
        if torch.cuda.is_available() and sync_cuda:
            self.sync_cuda = True
        else:
            self.sync_cuda = False
        self.start_time = time()

    def reset(self):
        self.start_time = time()
        return self

    def get_time(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
        return time() - self.start_time
