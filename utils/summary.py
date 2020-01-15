from collections import OrderedDict, Iterable
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = np.nan
        self.avg = np.nan
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else np.nan


class EvaluationMetrics(object):
    def __init__(self, keys=None):
        self.info = OrderedDict()
        if keys is not None:
            self.set(keys)

    def set(self, keys):
        assert isinstance(keys, Iterable), "keys must be iterable"
        for k in keys:
            self.info[k] = AverageMeter()

    def reset(self):
        for k, v in self.info.items():
            assert isinstance(v, AverageMeter), "data must be set first"
            self.info[k].reset()

    def update(self, key, value, n=1):
        assert key in self.info.keys(), "key does not exist"
        self.info[key].update(value, n)

    @property
    def val(self):
        info = OrderedDict()
        for k, v in self.info.items():
            info[k] = v.val
        return info

    @property
    def sum(self):
        info = OrderedDict()
        for k, v in self.info.items():
            info[k] = v.sum
        return info

    @property
    def avg(self):
        info = OrderedDict()
        for k, v in self.info.items():
            info[k] = v.avg
        return info
