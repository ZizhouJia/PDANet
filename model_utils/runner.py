import torch
from pynvml import *
import torch.nn as nn
from tensorboardX import SummaryWriter
import multiprocessing
import time
import signal
import traceback


#parallel function to send the models to certain device
def parallel(models, device_ids=[0]):
    #set device
    torch.cuda.set_device(device_ids[0])
    #set gpu mode
    for i in range(0,len(models)):
        models[i]=models[i].cuda()
    #sigle gpu
    if(len(device_ids)==1):
        return models
    #multiple GPU
    ret = []
    for i in range(0, len(models)):
        ret.append(nn.DataParallel(models[i], device_ids=device_ids))
    return ret


class worker(multiprocessing.Process):
    def __init__(self,t,device_use,error_dict,result_dict):
        multiprocessing.Process.__init__(self)
        self.t=t
        self.device_use=device_use
        self.error_dict=error_dict
        self.error_dict[t.task_name]=0
        self.result_dict=result_dict
        self.result_dict[t.task_name]=-1

    def run(self):
        try:
            solver,solver_param_dict=self._init_task(self.t,self.device_use)
            self.result_dict[self.t.task_name]=solver.main(solver_param_dict)
        except Exception as e:
            traceback.print_exc()
            self.result_dict[self.t.task_name]="error"

    def _init_task(self,t,device_use):
        #build models
        models=[]
        for i in range(0,len(t.models)):
            model_class=t.models[i]["class"]
            param=t.models[i]["params"]
            models.append(model_class(**param))
        models=parallel(models,device_use)
        optimizers=t.optimizers_producer["function"](models,**t.optimizers_producer["params"])
        kernel_processer=t.kernel_processer["class"](**t.kernel_processer["params"])
        solver=t.solver["class"](models,optimizers,kernel_processer,t.task_name,"checkpoints",**t.solver["params"])
        solver_param_dict=t.dataset_producer["function"](**t.dataset_producer["params"])
        return solver,solver_param_dict


class task(object):
    def __init__(self):
        self.task_name=None
        self.solver=None
        self.models=None
        self.optimizers_producer=None
        self.kernel_processer_class=None
        self.dataset_producer=None
        self.memory_use=[]

#a runner is a controller of a set of tasks and a set of gpus
class runner(object):
    def __init__(self):
        self.writer=SummaryWriter("runs/runner-logs")
        nvmlInit()
        self.nvidia_free=[]
        self.nvidia_total=[]
        self.tasks=[]
        self.running_tasks=[]
        self.manager=multiprocessing.Manager()
        self.error_dict=self.manager.dict()
        self.result_dict=self.manager.dict()

    def update_nvidia_info(self):
        nvidia_free=[]
        nvidia_total=[]
        deviceCount = nvmlDeviceGetCount()
        for i in range(0,deviceCount):
            handle=nvmlDeviceGetHandleByIndex(i)
            meminfo=nvmlDeviceGetMemoryInfo(handle)
            nvidia_free.append(meminfo.free/1024/1024)
            nvidia_total.append(meminfo.total/1024/1024)
        self.nvidia_free=nvidia_free
        self.nvidia_total=nvidia_total

    def generate_tasks(self,task_list):
        tasks=[]
        for i in range(0,len(task_list)):
            t=task()
            t.task_name=task_list[i]["task_name"]
            t.solver=task_list[i]["solver"]
            t.models=task_list[i]["models"]
            t.optimizers_producer=task_list[i]["optimizers"]
            t.kernel_processer=task_list[i]["kernel"]
            t.memory_use=task_list[i]["mem_use"]
            t.dataset_producer=task_list[i]["dataset"]
            tasks.append(t)
        self.tasks=tasks

    def _dispatch_cards(self,mem_use):
        card_use=[]
        for i in range(0,len(mem_use)):
            mem=mem_use[i]
            for j in range(0,len(self.nvidia_free)):
                if(j in card_use or mem>self.nvidia_free[j]):
                    continue
                else:
                    card_use.append(j)
                    break
        if(len(card_use)==len(mem_use)):
            return card_use
        else:
            return -1

    def main_loop(self):
        # multiprocessing.set_start_method("spawn")
        while(len(self.tasks)!=0 or len(self.running_tasks)!=0):
            self.update_nvidia_info()
            handled=-1
            for i in range(0,len(self.tasks)):
                t=self.tasks[i]
                device_use=self._dispatch_cards(t.memory_use)
                if(device_use!=-1):
                    print("************************begin task "+t.task_name+"****************")
                    w=worker(t,device_use,self.error_dict,self.result_dict)
                    w.start()
                    self.running_tasks.append(t.task_name)
                    handled=i
                    time.sleep(20)
                    break
            if(handled!=-1):
                del self.tasks[handled]
            else:
                print("no device can be used")
                time.sleep(10)
            end_name=-1
            for i in range(0,len(self.running_tasks)):
                t_name=self.running_tasks[i]
                if(self.result_dict[t_name]!=-1):
                    end_name=i
                    print("************************end task "+t_name+"****************")
                    break;
                if(self.result_dict[t_name]=="error"):
                    end_name=i
                    print(self.error_dict[t_name])
                    break;
            if(end_name!=-1):
                del self.running_tasks[end_name]
