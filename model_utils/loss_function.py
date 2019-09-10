import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def D_classify_loss(output, target):
    e = 1e-8
    # loss=target*torch.log(output)+(1-target)*torch.log(1-output)
    output = torch.exp(output)
    total = torch.sum(output, 1).view(output.size()[0], 1)
    output = torch.log(output/total+e)
    # print(target)
    # print(output)
    output = torch.sum(-target*output, 1)
    return output.mean()


def G_classify_loss(output):
    e = 1e-8
    output = torch.exp(output)
    total = torch.sum(output, 1).view(output.size()[0], 1)
    output = torch.log(output/total+e)
    output = torch.mean(-output, 1)
    return output.mean()

# loss function


def l1_loss(x1, x2):
    loss = torch.abs(x1-x2).mean()
    return loss


def D_real_loss(output, loss_func='lsgan'):
    if(loss_func == 'lsgan'):
        distance = (output-1.0)*(output-1.0)
        loss = distance.mean()
        return loss

    if(loss_func == 'wgan'):
        return (-output).mean()

    if(loss_func == 'hinge'):
        real_loss = torch.functional.F.relu(1.0 - output).mean()
        return real_loss


def D_fake_loss(output, loss_func='lsgan'):
    if(loss_func == 'lsgan'):
        distance = output*output
        loss = distance.mean()
        return loss

    if(loss_func == 'wgan'):
        return output.mean()

    if(loss_func == 'hinge'):
        real_loss = torch.functional.F.relu(1.0 + output).mean()
        return real_loss


def G_fake_loss(output, loss_func='lsgan'):
    if(loss_func == 'lsgan'):
        distance = (output-1)*(output-1)
        loss = distance.mean()
        return loss

    if(loss_func == 'wgan'):
        return (-output).mean()

    if(loss_func == 'hinge'):
        return (-output).mean()


def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda()
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)
