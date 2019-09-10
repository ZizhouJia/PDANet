import torch
import model_utils.common_solver as c_solver
import net4 as net
import regression_core
import WSCNet_core
import dataset
import model_utils.runner as runner

r=runner.runner()

task_midrnet_IAPS={
"task_name":"baseline_mdirnet_IAPS_e300_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{}},
"models":[{"class":net.mldr_net,"params":{}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.001,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./IAPS","dataset":"IAPS","batch_size":64,"epochs":300}},
"mem_use":[10000,10000,10000,10000]
}

task_midrnet_NAPS={
"task_name":"baseline_mdirnet_NAPS_e300_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{}},
"models":[{"class":net.mldr_net,"params":{}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.001,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./NAPS","dataset":"NAPS","batch_size":64,"epochs":300}},
"mem_use":[10000,10000,10000,10000]
}

task_midrnet_EMOTIC={
"task_name":"baseline_mdirnet_EMOTIC_e50_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{}},
"models":[{"class":net.mldr_net,"params":{}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.001,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./EMOTIC","dataset":"EMOTIC","batch_size":64,"epochs":50}},
"mem_use":[10000,10000,10000,10000]
}

r.generate_tasks([task_midrnet_NAPS])
r.main_loop()
