import argparse
import random
import string
import logging
import time
import json
from pynvml import *
from flask import Flask,request
from flask_apscheduler import APScheduler

nvidia_total=[]
nvidia_free=[]

def produce_random_string():
    return ".join(random.sample(string.ascii_letters+string.digits,8))


def get_time_string():
    dt = datetime.now()
    return dt.strftime("%Y%m%d%H%M")


def update_nvidia_info():
    nvidia_free=[]
    nvidia_total=[]
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle=nvmlDeviceGetHandleByIndex(i)
        meminfo=nvmlDeviceGetMemoryInfo(handle)
        nvidia_free.append(meminfo.free/1024/1024)
        nvidia_total.append(meminfo.total/1024/1024)


def dispatch_cards(gpu_memory_use,gpu_ids,ip=""):
    i=0
    for gpu_id in gpu_ids



class task_info:
    def __init__(self):
        self.task_name=None
        self.task_commit_time=None
        self.task_run_time=None
        self.task_lasttime_connect=None
        self.task_status=None
        self.ip=None
        self.key=None
        self.gpu_memory_use=None
        self.gpu_ids=None

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=str,help="the port of the task scheduler server")
    args=parser.parse_args()

    #set logger
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    rq=get_time_string()
    log_file="log_"+rq+".log"
    formatter=logging.Formatter("%(asctime)s - %(levelname)s : %(message)s")
    fh=logging.FileHandler(log_file,"w")
    fh.setFormatter(formatter)
    ch=logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    scheduler=APScheduler()
    writer=SummaryWriter("runs/task_logs")
    app=Flask(__name__)
    task_dick={}
    task_list=[]
    current_task=""
    root_passward="thu_thss"

    nvmlInit()

    @app.route('/register_task',methods=['POST'])
    def register_task():
        status=request.form['info']
        task_name=request.form['task_name']
        if(status=="WAIT"):
            if(task_name not in task_list):
                task_list.append(task_name)
                t.task_name=task_name
                t.task_commit_time=get_time_string()
                t.task_lasttime_connect=time.now()
                t.key=produce_random_string()
                t.task_status="WAIT"
                t.ip=request.remote_addr
                t.gpu_memory_use=request.form.getlist["gpu_memory_use"]
                t.gpu_ids=request.form.getlist["gpu_ids"]
                task_dick[task_name]=t
                return json.dumps({"info":"REGISTER","key":t.key})
            else:
                t=task_dick[task_name]
                if(request.form["key"]!=t.key):
                    return json.dumps({"info":"ERROR"})
                if(current_task==task_name):
                    t.task_lasttime_connect=datetime.now()
                    gpu_ids=dispatch_cards(t.gpu_memory_use,t.gpu_ids)
                    if(len(gpu_ids)==0):
                        return json.dumps({"info":"WAIT"})
                    t.task_status="RUN"
                    return json.dumps({"info":"run","gpu_ids":gpu_ids})

        if(status=="RUN"):
            if(task_name not in task_list):
                return json.dumps({"info":"STOP"})
            t=task_dick[task_name]
            if(t.task_status!="RUN"):
                return json.dumps({"info":"ERROR"})
            t.task_lasttime_connect=time.now()
            return json.dumps({"info":"OK"})

        if(status=="STOP"):
            if(task_name in task_list):
                task_list.remove(task_name)
                del task_dick


    @app.route('./ask_stop',methods=['POST'])
    def ask_stop():
        task_name=request.args["task_name"]
        passward=request.args["passward"]
        if(passward=="thuss"):
            if(task_name in task_list):
                task_list.remove(task_name)
                del task_dick
                return "Succeed"
            else:
                return "The task not exists"
        else:
            return "the passward is wrong"


    @app.route('/',methods=['GET','POST'])
    def index():
        return_dict={}
        return_list=[]
        for i in range(0,len(task_list)):
            t=task_dick[task_list[i]]
            task_dict={}
            task_dick["task_name"]=t.task_name
            task_dick["task_commit_time"]=t.task_commit_time
            task_dick["task_run_time"]=t.task_run_time
            task_dict["task_status"]=t.task_status
            task_dick["ip_address"]=str(t.ip)
            task_dict["gpu_memory_use"]=t.gpu_memory_use
            task_dick["gpu_ids"]=t.gpu_ids
            return_list.append(task_dick)
        return_dict["task list"]=return_list
        return json.dumps(return_dict)

    def time_task():
        #check the TIME_OUT
        for task_name in task_list:
            t=task_dick[task_name]
            if(t.status not in [task_signal.TIME_OUT,task_signal.PRE_STOP,task_signal.STOP]):
                if((datetime.now()-t.task_lasttime_connect).seconds>=10):
                    t.status=task_signal.TIME_OUT

        #dispatch the card
        for task_name in task_list:
            t=task_dick[task_name]
            if(t.status==task_signal.WAIT):
                if(len(dispatch_cards(t.gpu_memory_use,t.gpu_ids))!=0):
                    current_task=task_name
                    break;
