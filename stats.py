import numpy as np
from time import time
from torch.utils.tensorboard import SummaryWriter


class Stats(object):
    def __init__(self, keys, log_dir=None, print_step=100, prefix=None):
        self.keys = keys
        self.summary_writer = SummaryWriter(log_dir)
        self.print_step = print_step
        self.prefix = prefix
        self.sums = {k: .0 for k in keys}
        self.start_time = time()
        self.count = 0

    def clear(self):
        for key in self.sums:
            self.sums[key] = .0
        self.start_time = time()
        self.count = 0

    def update(self, *args):
        for key, val in zip(self.keys, args):
            self.sums[key] += float(val)
        self.count += 1

    def summarize(self, step):
        stats = dict.fromkeys(self.sums)
        for key in self.sums:
            stats[key] = self.sums[key] / self.count
            tag = key if self.prefix is None else self.prefix + key
            self.summary_writer.add_scalar(tag, stats[key], step)
        time_ms = int(np.round(1e3 * (time() - self.start_time)) / self.count)
        return stats, time_ms

    def pretty_print(self, step, stats, time_ms):
        step_str = ['{:<8}'.format(str(step) + ')')]
        stats_str = ['{}: {:<9.4f}'.format(k, stats[k]) for k in self.keys]
        time_str = ['{:>10}'.format('(' + str(time_ms) + ' msec)')]
        str_out = ' '.join(step_str + stats_str + time_str)
        print(str_out)

    def __call__(self, step, *args):
        self.update(*args)
        if (step + 1) % self.print_step == 0:
            stats, time_ms = self.summarize(step)
            self.clear()
            self.pretty_print(step + 1, stats, time_ms)