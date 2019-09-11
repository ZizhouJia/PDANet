import model_utils.kernel_processer as k_processor
import torch
import torch.nn.functional as F
import numpy as np
import math

class visual_processer(k_processor.kernel_processer):
    def __init__(self):
        super(visual_processer, self).__init__()

    def train(self,step,data):
        pass


    def evaluate(self,step,data):
        data=self.tencrop_process(data)
        model=self.models[0]
        evaluate_dict={}
        x=data[0]
        y=data[1]
        out,v,M=model(x)
        x_m=M.size(2)
        y_m=M.size(3)
        M=M.view(-1,1,x_m*y_m)
        M=M.view(-1,1,x_m,y_m)
        return y.detach().cpu().numpy(),out.detach().cpu().numpy(),x.detach().cpu().numpy(),M.detach().cpu().numpy()


def optimizers_producer(models,lr_base,lr_fc,weight_decay,paral=True):
    optimizers=[]
    model=models[0]
    c_param=None
    if(paral):
        c_param=model.module.fc.parameters()
    else:
        c_param=model.fc.parameters()

    clssify_params = list(map(id, c_param))
    base_params = filter(lambda p: id(p) not in clssify_params,model.parameters())
    optimizer = torch.optim.SGD([
        {'params': base_params,'lr':lr_base},
        {'params': c_param, 'lr': lr_fc},
        ], lr_base, momentum=0.9, weight_decay=weight_decay)
    optimizers.append(optimizer)
    return optimizers


def optimizers_producer_classify(models,lr_base,lr_fc,weight_decay,paral=True):
    optimizers=[]
    model=models[0]
    c_param=None
    classify_param=None
    if(paral):
        c_param=model.module.fc.parameters()
        classify_param=model.module.fc_classify.parameters()
    else:
        c_param=model.fc.parameters()
        classify_param=model.fc_classify.parameters()

    c_params = list(map(id, c_param))
    classify_params=list(map(id,classify_param))
    base_params = filter(lambda p: id(p) not in classify_params+c_params,model.parameters())
    optimizer = torch.optim.SGD([
        {'params': base_params,'lr':lr_base},
        {'params': c_param, 'lr': lr_fc},
        {'params': classify_param, 'lr': lr_fc},
        ], lr_base, momentum=0.9, weight_decay=weight_decay)
    optimizers.append(optimizer)
    return optimizers
