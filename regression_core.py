import model_utils.kernel_processer as k_processor
import torch
import torch.nn.functional as F
import numpy as np
import math

class regression_processer(k_processor.kernel_processer):
    def __init__(self,grad_clip=False,clip_value=5.0,epoch_lr_decay=250,
    classify_mode=False,loss_constrain=False,constrain_value=1.0):
        super(regression_processer, self).__init__()
        self.loss_function=torch.nn.MSELoss()
        self.bce_loss=torch.nn.BCELoss()
        self.grad_clip=grad_clip
        self.clip_value=clip_value
        self.epoch_lr_decay=epoch_lr_decay
        self.classify_mode=classify_mode
        self.loss_constrain=loss_constrain
        self.constrain_value=constrain_value
        self.pred=None
        self.label=None

    def train(self,step,data):
        optimizer=self.optimizers[0]
        model=self.models[0]
        total_loss={}
        x=data[0]
        y=data[1]
        out,v,M=model(x)

        loss_pred=self.loss_function(out,y)
        loss=None
        if(len(out.size())!=len(M.size())):
            loss=self.loss_function(v,y)+loss_pred
        else:
            if(self.classify_mode):
                label_c=(y>0.5).float()
                out_logit=(out>0.5).float()
                mask=1-(label_c==out_logit).float()
                loss=loss_pred+self.constrain_value*torch.mean(mask*(out-y)*(out-y))
            else:
                loss=loss_pred
        loss.backward()
        optimizer.step()
        self.zero_grad_for_all()
        mse=(out-y)*(out-y)
        mse=torch.mean(mse,0)
        mse=mse.detach().cpu().numpy()
        if(step%10==0):
            total_loss["train_loss"]=loss.detach().cpu().item()
            total_loss["train_loss_pred"]=loss_pred.detach().cpu().item()
        return total_loss

    def evaluate(self,step,data):
        data=self.tencrop_process(data)
        model=self.models[0]
        evaluate_dict={}
        x=data[0]
        y=data[1]
        out,v,M=model(x)
        # out=F.sigmoid(out)
        loss_pred=self.loss_function(out,y)
        loss=None
        if(len(out.size())!=len(M.size())):
            loss=self.loss_function(v,y)+loss_pred
        else:
            if(self.classify_mode):
                label_c=(y>0.5).float()
                out_logit=(out>0.5).float()
                mask=1-(label_c==out_logit).float()
                loss=loss_pred+self.constrain_value*torch.mean(mask*(out-y)*(out-y))
            else:
                loss=loss_pred
        mse=(out-y)*(out-y)
        mse=torch.mean(mse,0)
        mse=mse.detach().cpu().numpy()

        evaluate_dict["test_loss"]=loss.detach().cpu().item()
        evaluate_dict["test_loss_pred"]=loss_pred.detach().cpu().item()
        return x.size(0),evaluate_dict["test_loss_pred"],evaluate_dict

    def test(self,step,data):
        data=self.tencrop_process(data)
        model=self.models[0]
        evaluate_dict={}
        x=data[0]
        y=data[1]
        out,v,M=model(x)
        # out=F.sigmoid(out)
        loss_pred=self.loss_function(out,y)
        loss=None
        if(len(out.size())!=len(M.size())):
            loss=self.loss_function(v,y)+loss_pred
        else:
            if(self.classify_mode):
                label_c=(y>0.5).float()
                out_logit=(out>0.5).float()
                mask=1-(label_c==out_logit).float()
                loss=loss_pred+self.constrain_value*torch.mean(mask*(out-y)*(out-y))
            else:
                loss=loss_pred
        mse=(out-y)*(out-y)
        mse=torch.mean(mse,0)
        mse=mse.detach().cpu().numpy()
        if(self.pred is None):
            self.pred=out.detach().cpu().numpy()
            self.label=y.detach().cpu().numpy()
        else:
            self.pred=np.concatenate((self.pred,out.detach().cpu().numpy()),axis=0)
            self.label=np.concatenate((self.label,y.detach().cpu().numpy()),axis=0)

        evaluate_dict["test_loss"]=loss.detach().cpu().item()
        evaluate_dict["test_loss_pred"]=loss_pred.detach().cpu().item()
        return x.size(0),evaluate_dict["test_loss_pred"],evaluate_dict

    def on_finish(self):
        d={}
        mse=(self.pred-self.label)*(self.pred-self.label)
        mse=np.mean(mse,axis=0)
        d["mse_1"]=mse[0]
        d["mse_2"]=mse[1]
        d["mse_3"]=mse[2]
        d["mse_mean"]=(mse[0]+mse[1]+mse[2])/3
        d["rmse_1"]=math.sqrt(mse[0])
        d["rmse_2"]=math.sqrt(mse[1])
        d["rmse_3"]=math.sqrt(mse[2])
        d["rmse_mean"]=(d["rmse_1"]+d["rmse_2"]+d["rmse_3"])/3
        mae=np.abs(self.pred-self.label)
        mae=np.mean(mae,axis=0)
        d["mae_1"]=mae[0]
        d["mae_2"]=mae[1]
        d["mae_3"]=mae[2]
        d["mae_mean"]=(mae[0]+mae[1]+mae[2])/3
        avg_label=np.mean(self.label,axis=0).reshape((1,3))
        mse_y=(self.label-avg_label)*(self.label-avg_label)
        mse_y=np.mean(mse_y,axis=0)
        r_sq=1-mse/mse_y
        d["r_square_1"]=r_sq[0]
        d["r_square_2"]=r_sq[1]
        d["r_square_3"]=r_sq[2]
        d["r_square_mean"]=np.mean(r_sq)
        return d


    def update_optimizers(self,epoch,step,total_data_numbers):
        optimizer=self.optimizers[0]
        if(epoch==self.epoch_lr_decay and step==0):
            print("change the learning rate ")
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']*0.1



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
