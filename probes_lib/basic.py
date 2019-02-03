from .top import *


class AccLossProbe(StatsProbe):
    def __init__(self, **kwargs):
        super(AccLossProbe, self).__init__()
        self.type = kwargs["type"]
        assert(self.type == "train" or self.type == "test")
        self.last_epoch_stats = {}

    def get_last_epoch_stats(self):
        return self.last_epoch_stats

    def epoch_prologue(self):
        self.last_epoch_stats = {}

    def add_data(self, **kwargs):
        loss_key = self.type + "_loss"
        acc_key = self.type + "_acc"
        self.last_epoch_stats[loss_key] = kwargs[loss_key]
        self.last_epoch_stats[acc_key] = kwargs[acc_key]


class AvgAccOverTasksProbe(StatsProbe):
    def __init__(self, **kwargs):
        super(AvgAccOverTasksProbe, self).__init__()
        self.type = kwargs["type"]
        assert(self.type == "train" or self.type == "test")
        self.last_epoch_stats = {}

    def get_last_epoch_stats(self):
        return self.last_epoch_stats

    def epoch_prologue(self):
        self.last_epoch_stats = {}

    def add_data(self, **kwargs):
        self.last_epoch_stats["avg_acc_over_tasks"] = kwargs["avg_acc_over_tasks"]


class EpochNumProbe(StatsProbe):
    def __init__(self):
        super(EpochNumProbe, self).__init__()
        self.last_epoch_stats = {}

    def get_last_epoch_stats(self):
        return self.last_epoch_stats

    def epoch_prologue(self):
        self.last_epoch_stats = {}

    def add_data(self, **kwargs):
        self.last_epoch_stats["epoch"] = kwargs["epochs_trained"]


class TimingProbe(StatsProbe):
    def __init__(self):
        super(TimingProbe, self).__init__()
        self.last_epoch_stats = {}

    def get_last_epoch_stats(self):
        return self.last_epoch_stats

    def epoch_prologue(self):
        self.last_epoch_stats = {}

    def add_data(self, **kwargs):
        self.last_epoch_stats["avg_time"] = kwargs.get("avg_time", 0)
        self.last_epoch_stats["avg_n"] = kwargs.get("avg_n", 0)


class IterationNumProbe(StatsProbe):
    def __init__(self):
        super(IterationNumProbe, self).__init__()
        self.num_of_iterations = 0
        self.last_iteration_stats = {}

    def get_last_iteration_stats(self):
        return self.last_iteration_stats

    def iteration_prologue(self):
        self.last_iteration_stats = {}

    def add_data(self, **kwargs):
        if self.training:
            self.last_iteration_stats["iteration"] = self.num_of_iterations
            self.num_of_iterations += 1