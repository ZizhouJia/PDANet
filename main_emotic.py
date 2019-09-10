import torch
import model_utils.common_solver as c_solver
import net4 as net
import regression_core
import WSCNet_core
import dataset
import model_utils.runner as runner

r=runner.runner()

task_baseline_EMOTIC={
"task_name":"baseline_EMOTIC_e50_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":40}},
"models":[{"class":net.Baseline,"params":{"C":3}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./EMOTIC","dataset":"EMOTIC","batch_size":32,"epochs":50}},
"mem_use":[10000,10000]
}

task_WSCNet_EMOTIC={
"task_name":"WSCNet_EMOTIC_e50_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":40}},
"models":[{"class":net.WSCNet,"params":{"k":4,"C":3}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./EMOTIC","dataset":"EMOTIC","batch_size":32,"epochs":50}},
"mem_use":[10000,10000]
}

task_sentiNet_EMOTIC={
"task_name":"sentiNet_EMOTIC_e25_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":20}},
"models":[{"class":net.SENet_senti_attention_wise,"params":{"C":3}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./EMOTIC","dataset":"EMOTIC","batch_size":32,"epochs":25}},
"mem_use":[10000,10000]
}

task_attention_wise_EMOTIC={
"task_name":"SENet_attention_wise_EMOTIC_e50_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":40}},
"models":[{"class":net.SENet_attention_wise,"params":{"C":3,"activate_type":"sigmoid_res"}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./EMOTIC","dataset":"EMOTIC","batch_size":32,"epochs":50}},
"mem_use":[10000,10000]
}

task_channel_wise_EMOTIC={
"task_name":"SENet_channel_wise_EMOTIC_e50_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":40}},
"models":[{"class":net.SENet_channel_wise,"params":{"C":3,"activate_type":"sigmoid_res"}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./EMOTIC","dataset":"EMOTIC","batch_size":32,"epochs":50}},
"mem_use":[10000,10000]
}

task_SENet_channel_wise_attention_EMOTIC={
"task_name":"SENet_channel_wise_attention_EMOTIC_e50_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":40}},
"models":[{"class":net.SENet_channel_wise_with_attention,"params":{"C":3,"activate_type":"sigmoid_res"}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./EMOTIC","dataset":"EMOTIC","batch_size":32,"epochs":50}},
"mem_use":[10000,10000]
}

task_SENet_channel_wise_attention_classify_EMOTIC={
"task_name":"SENet_channel_wise_attention_classify_EMOTIC_e50_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":40,"classify_mode":True}},
"models":[{"class":net.SENet_channel_wise_with_attention,"params":{"C":3,"activate_type":"sigmoid_res"}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./EMOTIC","dataset":"EMOTIC","batch_size":32,"epochs":50}},
"mem_use":[10000,10000]
}

task_SENet_channel_wise_attention_classify_constrain_EMOTIC={
"task_name":"SENet_channel_wise_attention_classify_constrain_EMOTIC_e25_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":20,"classify_mode":True,"loss_constrain":True}},
"models":[{"class":net.SENet_channel_wise_with_attention,"params":{"C":3,"activate_type":"sigmoid_res"}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./EMOTIC","dataset":"EMOTIC","batch_size":32,"epochs":25}},
"mem_use":[10000,10000]
}


task_midrnet_IAPS={
"task_name":"baseline_mdirnet_EMOTIC_e300_experiment",
"solver":{"class":c_solver.common_solver,"params":{}},
"kernel":{"class":regression_core.regression_processer,"params":{"epoch_lr_decay":20}},
"models":[{"class":net.EmoNet,"params":{}}],
"optimizers":{"function":regression_core.optimizers_producer,"params":{"lr_base":0.1,"lr_fc":0.1,"weight_decay":0.0005,"paral":True}},
"dataset":{"function":dataset.param_dict_producer,"params":{"path":"./EMOTIC","dataset":"./EMOTIC","batch_size":32,"epochs":25}},
"mem_use":[10000,10000]
}

tasks=[]

# tasks.append(task_baseline_EMOTIC)
tasks.append(task_WSCNet_EMOTIC)
# # tasks.append(task_sentiNet_EMOTIC)
# tasks.append(task_attention_wise_EMOTIC)
# tasks.append(task_channel_wise_EMOTIC)
tasks.append(task_SENet_channel_wise_attention_EMOTIC)
tasks.append(task_SENet_channel_wise_attention_classify_EMOTIC)
# tasks.append(task_midrnet_IAPS)
# tasks.append(task_SENet_channel_wise_attention_classify_constrain_EMOTIC)

r.generate_tasks(tasks)
r.main_loop()
