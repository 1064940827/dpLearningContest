import torch
import numpy as np
from thop import profile
from thop import clever_format
from torch import optim

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, decay_epoch=100):
    # 弃用
    lr_lambda = lambda epoch: 1.0 - pow((epoch / decay_epoch), 0.9)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_scheduler.get_last_lr()[0]


class AvgMeter(object):
    # 计算num个图像的loss平均值
    def __init__(self, num=40):
        self.num = num  # 用来指定保留计算的最后 num 个值
        self.reset()

    def reset(self):
        self.val = 0  # 当前的损失值，update 方法传入的 val 参数
        self.avg = 0  # 所有传入的值的累计平均值
        self.sum = 0  # 所有传入的值的累加和
        self.count = 0 # 已传入的值的总数
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))