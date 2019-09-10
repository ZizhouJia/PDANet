import torch
import torch.nn as nn

class kernel_processer(object):
    def __init__(self):
        self.models=None
        self.optimizers=None

    def set_models(self,models):
        self.models=models

    def set_optimizers(self,optimizers):
        self.optimizers=optimizers

    def tencrop_process(self,data):
        x=data[0]
        if(len(x.size())==5):
            x=data[0]
            y=data[1]
            crop_size=x.size(1)
            list_y=[]
            size_y=list(y.size())
            size_y[0]=size_y[0]*crop_size
            y=torch.unsqueeze(y,1)
            for i in range(0,crop_size):
                list_y.append(y)
            y=torch.cat(list_y,1)
            y=y.view(size_y)
            x=x.view(-1,x.size(2),x.size(3),x.size(4))
            data=(x,y)
        return data

    def update_optimizers(self,epoch,step,total_data_numbers):
        pass

    def on_finish(self):
        pass

    def zero_grad_for_all(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def train(self,step,data):
        raise NotImplementedError

    def test(self,step,data):
        raise NotImplementedError

    def evaluate(self,step,data):
        raise NotImplementedError

    def update_optimizers(self,epoch,step,total_data_numbers):
        pass
