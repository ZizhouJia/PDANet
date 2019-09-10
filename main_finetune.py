# -*- coding: UTF-8 -*-
#对 iaps naps emotic's vgg_net vggnet
import torch
import model_utils.common_solver as c_solver
import net2 as net
import regression_core
import WSCNet_core
import dataset
import model_utils.runner as runner

r=runner.runner()


task_baseline_alexnet_IAPS={
"task_name":"baseline_alexnet_IAPS_e300_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{}},
"models":[{"class":net.alex_net,"params":{"C":3}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":False}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./IAPS","dataset":"IAPS","batch_size":32,"epochs":300}},
"mem_use":[10000]
}

task_baseline_alexnet_NAPS={
"task_name":"baseline_alexnet_NAPS_e300_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{}},
"models":[{"class":net.alex_net,"params":{"C":3}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":False}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./NAPS","dataset":"NAPS","batch_size":32,"epochs":300}},
"mem_use":[10000]
}

task_baseline_alexnet_EMOTIC={
"task_name":"baseline_alexnet_EMOTIC_e25_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":20}},
"models":[{"class":net.alex_net,"params":{"C":3}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":False}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./EMOTIC","dataset":"EMOTIC","batch_size":32,"epochs":25}},
"mem_use":[10000]
}

task_baseline_vgg_net_IAPS={
"task_name":"baseline_vgg_net_IAPS_e300_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{}},
"models":[{"class":net.vgg_net,"params":{"C":3}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":False}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./IAPS","dataset":"IAPS","batch_size":32,"epochs":300}},
"mem_use":[10000]
}

task_baseline_vgg_net_NAPS={
"task_name":"baseline_vgg_net_NAPS_e300_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{}},
"models":[{"class":net.vgg_net,"params":{"C":3}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":False}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./NAPS","dataset":"NAPS","batch_size":32,"epochs":300}},
"mem_use":[10000]
}

task_baseline_vgg_net_EMOTIC={
"task_name":"baseline_vgg_net_EMOTIC_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":20}},
"models":[{"class":net.vgg_net,"params":{"C":3}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":False}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./EMOTIC","dataset":"EMOTIC","batch_size":32,"epochs":25}},
"mem_use":[10000]
}

tasks=[]
# tasks.append(task_baseline_alexnet_IAPS)
# tasks.append(task_baseline_vgg_net_IAPS)
# tasks.append(task_baseline_alexnet_NAPS)
# tasks.append(task_baseline_vgg_net_NAPS)
tasks.append(task_baseline_alexnet_EMOTIC)
tasks.append(task_baseline_vgg_net_EMOTIC)
r.generate_tasks(tasks)
r.main_loop()

