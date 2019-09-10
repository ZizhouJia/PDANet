from . import solver
import torch
import torch.nn as nn
import time

class common_solver(solver.solver):
    def __init__(self, models,optimizers,kernel_processer,model_name,save_path='checkpoints'):
        super(common_solver, self).__init__(
            models,optimizers,kernel_processer,model_name,save_path)

    def test_model(self, param_dict,mode="test"):
        loader_choice={"test":"test_loader","val":"val_loader"}
        self.eval_mode()
        dataloader=param_dict[loader_choice[mode]]
        counter=0.0
        evaluate_value=0.0
        evaluate_dict=None
        for step,data in enumerate(dataloader):
            for i in range(0,len(data)):
                data[i]=data[i].cuda()
            data_counter,key_value,output_dict=self.kernel_processer.test(step,data)
            counter+=data_counter
            evaluate_value+=(key_value*data_counter)
            if(evaluate_dict is None):
                evaluate_dict={}
                for key in output_dict.keys():
                    evaluate_dict[key]=output_dict[key]*data_counter
            else:
                for key in output_dict.keys():
                    evaluate_dict[key]+=(output_dict[key]*data_counter)
        for key in evaluate_dict.keys():
            evaluate_dict[key]=evaluate_dict[key]/counter
        evaluate_value=evaluate_value/counter
        return evaluate_value,evaluate_dict

    def evaluate_model(self,param_dict,mode="val"):
        loader_choice={"test":"test_loader","val":"val_loader"}
        self.eval_mode()
        dataloader=param_dict[loader_choice[mode]]
        counter=0.0
        evaluate_value=0.0
        evaluate_dict=None
        for step,data in enumerate(dataloader):
            for i in range(0,len(data)):
                data[i]=data[i].cuda()
            data_counter,key_value,output_dict=self.kernel_processer.evaluate(step,data)
            counter+=data_counter
            evaluate_value+=(key_value*data_counter)
            if(evaluate_dict is None):
                evaluate_dict={}
                for key in output_dict.keys():
                    evaluate_dict[key]=output_dict[key]*data_counter
            else:
                for key in output_dict.keys():
                    evaluate_dict[key]+=(output_dict[key]*data_counter)
        for key in evaluate_dict.keys():
            evaluate_dict[key]=evaluate_dict[key]/counter
        evaluate_value=evaluate_value/counter
        return evaluate_value,evaluate_dict


    def train_model(self,epoch,param_dict):
        self.train_mode()
        dataloader=param_dict["train_loader"]
        dataset_numbers=dataloader.dataset.__len__()
        it_numbers=int((dataset_numbers+dataloader.batch_size-1)/dataloader.batch_size)
        for step,data in enumerate(dataloader):
            for i in range(0,len(data)):
                data[i]=data[i].cuda()
            evaluate_dict=self.kernel_processer.train(step,data)
            self.write_log(evaluate_dict,epoch*it_numbers+step)
            self.output_loss(evaluate_dict,epoch,step)
            self.kernel_processer.update_optimizers(epoch,step,it_numbers)

    #param_dict ["train_loader","val_loader","test_loader"]
    def main(self,param_dict):
        best_value=1e10
        iteration_count=0
        epochs=param_dict["epochs"]
        for i in range(0,epochs):
            self.train_model(i,param_dict)
            evaluate_value,evaluate_dict=self.evaluate_model(param_dict,"val")
            self.output_loss(evaluate_dict,i,0)
            self.write_log(evaluate_dict,i)
            if(evaluate_value<best_value):
                best_value=evaluate_value
                self.save_params("best")
        self.restore_params(self.time_string,"best")
        tev,ted=self.test_model(param_dict,"test")
        self.write_log(ted,epochs+5)
        d=self.kernel_processer.on_finish()
        self.write_log(d,500)
        print(d)
        time.sleep(10)
        self.writer.close()
        return tev,ted
