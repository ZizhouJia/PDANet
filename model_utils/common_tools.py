import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# weight initialization


def weights_init(init_type='default'):
    def init_func(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
    return init_func


# optimizer
def generate_optimizers(models, lrs, optimizer_type='sgd', weight_decay=0.001):
    optimizers = []
    if(optimizer_type == 'sgd'):
        for i in range(0, len(models)):
            optimizer = torch.optim.SGD(models[i].parameters(
            ), lr=lrs[i], weight_decay=weight_decay, momentum=0.9)
            optimizers.append(optimizer)

    if(optimizer_type == 'adam'):
        for i in range(0, len(models)):
            optimizer = torch.optim.Adam(models[i].parameters(
            ), lr=lrs[i], weight_decay=weight_decay, betas=(0.5, 0.999))
            # optimizer=nn.DataParallel(optimizer)
            optimizers.append(optimizer)
    return optimizers


# parallel model
def parallel(models, device_ids=[0]):
    ret = []
    for i in range(0, len(models)):
        ret.append(nn.DataParallel(models[i], device_ids=device_ids))
    return ret
