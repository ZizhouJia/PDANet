import model_utils.kernel_processer as k_processor
import torch
import torch.nn.functional as F
import numpy as np
import math

class regression_processer(k_processor.kernel_processer):
    def __init__(self):
        super(regression_processer.self).__init__()
        self.loss_function=torch.nn.MSELoss()

    def train(self,step,data):


class regression_processer(k_processor.kernel_processer):
    def __init__(self):
        super(regression_processer, self).__init__()
        self.loss_function=torch.nn.MSELoss()

    def train(self,step,data):
        optimizer=self.optimizers[0]
        model=self.models[0]
        total_loss={}
        x=data[0]
        y=data[1]
        label_c=(y>0.5).float()
        index=label_c[:,0].numpy()*4+label_c[:,1].numpy()*2+label_c[:,2]
        label=torch.zeros((index.size(0),8))
        label[index]=1
        
        out,v,M=model(x)
        pred=F.sigmoid(out)
        loss=self.loss_function(pred,label_c)
        loss.backward()
        optimizer.step()
        self.zero_grad_for_all()
        total_loss["train_loss"]=loss.detach().cpu().item()
        return total_loss

    def evaluate(self,step,data):
        evaluate_dict={}
        data=self.tencrop_process(data)
        optimizer=self.optimizers[0]
        model=self.models[0]
        total_loss={}
        x=data[0]
        y=data[1]
        label_c=(y>0.5).float()
        out,out2,out3=model(x)
        pred=F.sigmoid(out)
        loss_pred=self.loss_function(pred,label_c)
        evaluate_dict["test_loss_pred"]=loss_pred.detach().cpu().item()
        return x.size(0),evaluate_dict["test_loss_pred"],evaluate_dict

    def test(self,step,data):
        return self.evaluate(step,data)


