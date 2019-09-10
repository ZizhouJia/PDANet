import model_utils.kernel_processer as k_processor
import torch


class WSCNet_kernel_processer(k_processor.kernel_processer):
    def __init__(self):
        super(WSCNet_kernel_processer, self).__init__()
        self.loss_function=torch.nn.CrossEntropyLoss()

    def train(self,step,data):
        optimizer=self.optimizers[0]
        model=self.models[0]
        x=data[0]
        y=data[1]
        out,v,M=model(x)
        pred=torch.max(out,1)[1]
        acc=torch.mean((pred==y).float())
        loss=self.loss_function(out,y)
        if(len(out.size())!=len(M.size())):
            loss+=self.loss_function(v,y)
        loss.backward()
        optimizer.step()
        self.zero_grad_for_all()
        total_loss={}
        total_loss["train_loss"]=loss.detach().cpu().item()
        total_loss["error_rate"]=1-acc.detach().cpu().item()
        return total_loss

    def test(self,step,data):
        data=self.tencrop_process(data)
        model=self.models[0]
        x=data[0]
        y=data[1]
        out,v,M=model(x)
        pred=torch.max(out,1)[1]
        loss=self.loss_function(out,y)
        if(len(out.size())!=len(M.size())):
            loss+=self.loss_function(v,y)
        correct=torch.mean((pred==y).float())
        evaluate_dict={}
        evaluate_dict["test_loss"]=loss.detach().cpu().item()
        evaluate_dict["test_error_rate"]=1-correct.detach().cpu().item()
        return x.size(0),evaluate_dict["test_error_rate"],evaluate_dict

    def evaluate(self,step,data):
        return self.test(step,data)

    def on_finish(self):
        return {}


    def update_optimizers(self,epoch,step,total_data_numbers):
        optimizer=self.optimizers[0]
        if((epoch==10 or epoch==20) and step==0):
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']*0.1
