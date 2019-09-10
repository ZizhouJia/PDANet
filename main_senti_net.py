import torch
import model_utils.common_solver as c_solver
import net2 as net
import regression_core
import WSCNet_core
import dataset
import model_utils.runner as runner

r=runner.runner()


task_sentiNet_IAPS={
"task_name":"sentiNet_IAPS",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{}},
"models":[{"class":net.SENet_senti_attention_wise,"params":{"C":3}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./IAPS","dataset":"IAPS","batch_size":32,"epochs":50}},
"mem_use":[10000,10000]
}

tasks=[]
tasks.append(task_sentiNet_IAPS)

r.generate_tasks(tasks)
r.main_loop()
