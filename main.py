# -*- coding: UTF-8 -*-
#对baseline wscnet sentinet spatial channel_wise spatial+channel_wise的IAPS和NAPS做实验
import torch
import model_utils.common_solver as c_solver
import net as net
import regression_core
import WSCNet_core
import dataset
import model_utils.runner as runner
import multiprocessing



if __name__ == '__main__':
    # multiprocessing.set_start_method("spawn",True)
    r=runner.runner()
    task_attention_wise_IAPS={
    "task_name":"SENet_attention_wise_IAPS_e300_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{"classify_mode":False}},
    "models":[{"class":net.SENet_attention_wise,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./IAPS","dataset":"IAPS","batch_size":32,"epochs":300}},
    "mem_use":[10000,10000]
    }

    task_attention_wise_NAPS={
    "task_name":"SENet_attention_wise_NAPS_e300_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{"classify_mode":False}},
    "models":[{"class":net.SENet_attention_wise,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./NAPS","dataset":"NAPS","batch_size":32,"epochs":300}},
    "mem_use":[10000,10000]
    }

    task_channel_wise_IAPS={
    "task_name":"SENet_channel_wise_IAPS_e300_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{}},
    "models":[{"class":net.SENet_channel_wise,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./IAPS","dataset":"IAPS","batch_size":32,"epochs":300}},
    "mem_use":[10000,10000]
    }

    task_channel_wise_NAPS={
    "task_name":"SENet_channel_wise_NAPS_e300_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{}},
    "models":[{"class":net.SENet_channel_wise,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./NAPS","dataset":"NAPS","batch_size":32,"epochs":300}},
    "mem_use":[10000,10000]
    }


    task_SENet_channel_wise_attention_IAPS={
    "task_name":"SENet_channel_wise_attention_IAPS_e300_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{}},
    "models":[{"class":net.SENet_channel_wise_with_attention,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./IAPS","dataset":"IAPS","batch_size":32,"epochs":300}},
    "mem_use":[10000,10000]
    }

    task_SENet_channel_wise_attention_NAPS={
    "task_name":"SENet_channel_wise_attention_NAPS_e300_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{}},
    "models":[{"class":net.SENet_channel_wise_with_attention,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./NAPS","dataset":"NAPS","batch_size":32,"epochs":300}},
    "mem_use":[10000,10000]
    }

    task_SENet_channel_wise_attention_classify_IAPS={
    "task_name":"SENet_channel_wise_attention_classify_IAPS_e300_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{"classify_mode":True}},
    "models":[{"class":net.SENet_channel_wise_with_attention,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer_classify,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./IAPS","dataset":"IAPS","batch_size":32,"epochs":300}},
    "mem_use":[10000,10000]
    }


    task_SENet_channel_wise_attention_classify_NAPS={
    "task_name":"SENet_channel_wise_attention_classify_NAPS_e300_10_experiment",
    "solver":{"class":c_solver.common_solver,"params":{}},
    "kernel":{"class":regression_core.regression_processer,"params":{"classify_mode":True,"constrain_value":10}},
    "models":[{"class":net.SENet_channel_wise_with_attention,"params":{"C":3,"activate_type":"sigmoid_res"}}],
    "optimizers":{"function":regression_core.optimizers_producer_classify,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
    "dataset":{"function":dataset.param_dict_producer,"params":{"path":"./NAPS","dataset":"NAPS","batch_size":32,"epochs":300}},
    "mem_use":[10000,10000]
    }


    tasks=[]

    tasks.append(task_attention_wise_IAPS)
    tasks.append(task_attention_wise_NAPS)
    tasks.append(task_channel_wise_IAPS)
    tasks.append(task_channel_wise_NAPS)
    tasks.append(task_SENet_channel_wise_attention_IAPS)
    tasks.append(task_SENet_channel_wise_attention_NAPS)
    tasks.append(task_SENet_channel_wise_attention_classify_IAPS)
    tasks.append(task_SENet_channel_wise_attention_classify_NAPS)

r.generate_tasks(tasks)
r.main_loop()
