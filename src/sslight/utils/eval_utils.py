import torch
import numpy as np
from collections import deque


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        if self.count == 0:
            return str(self.val)

        return f'{self.val:.4f} ({self.avg:.4f})'


class MovingAverageMeter(object):
    def __init__(self, window):
        self.window = window
        self.reset()

    def reset(self):
        self.history = deque()
        self.avg = 0
        self.sum = None
        self.val = None

    @property
    def count(self):
        return len(self.history)

    @property
    def isfull(self):
        return len(self.history) == self.window

    def __getstate__(self):
        state = self.__dict__.copy()
        state['history'] = np.array(state['history'])
        return state

    def __setstate__(self, state):
        state['history'] = deque(state['history'])
        self.__dict__.update(state)

    def update(self, val, n=1):
        if n == 1:
            self.update_one_sample(val)
            return 

        self.history.extend([val] * n)
        if self.sum is None:
            self.sum = val * n
        else:
            self.sum += val * n
        while len(self.history) > self.window:
            self.sum -= self.history.popleft()
        self.val = val
        self.avg = self.sum / self.count
    
    def update_one_sample(self, val):
        self.history.append(val)
        if self.sum is None:
            self.sum = val
        else:
            self.sum += val
        if len(self.history) > self.window:
            self.sum -= self.history.popleft()
        self.val = val
        self.avg = self.sum / self.count

    def __str__(self):
        if self.count == 0:
            return str(self.val)

        return f'{self.val:.4f} ({self.avg:.4f})'

    def __repr__(self):
        return "<MovingAverageMeter of window {} with {} elements, val {}, avg {}>".format(
            self.window, self.count, self.val, self.avg)