# -*- coding: UTF-8 -*-
#对 classfy 单独实验
import torch
import model_utils.common_solver as c_solver
import net
import classify_core
import regression_core
import WSCNet_core
import dataset
import model_utils.runner as runner

r=runner.runner()


task_baseline_classify={
"task_name":"baseline_classify_FI",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":WSCNet_core.WSCNet_kernel_processer,"params":{}},
"models":[{"class":net.Baseline,"params":{"C":8}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./FI","dataset":"FI","batch_size":32,"epochs":40}},
"mem_use":[10000,10000]
}

r.generate_tasks([task_baseline_classify])
r.main_loop()
