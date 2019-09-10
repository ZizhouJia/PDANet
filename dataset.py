import numpy as np
from torchvision import transforms
import model_utils.common_dataset as cd
import torch.utils.data as Data
import FI_dataset
import os
import torch
import math


resize_transforms={
'train': transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'val': transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'test': transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

}

pretrain_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ])
}

pretrain_transforms_test = {
    'train': transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    'test': transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
}


resnet_transforms = {
    'train': transforms.Compose([
        transforms.Resize(600),
        transforms.RandomResizedCrop(446),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(600),
        transforms.TenCrop(446),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ]),
    'test': transforms.Compose([
        transforms.Resize(600),
        transforms.TenCrop(446),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ])
}

mldr_transforms = {
    'train': transforms.Compose([
        transforms.Resize(400),
        transforms.RandomResizedCrop(375),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(400),
        transforms.TenCrop(375),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ]),
    'test': transforms.Compose([
        transforms.Resize(400),
        transforms.TenCrop(375),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ])
}


def read_txt(path,div=8.0):
    image_dict={}
    with open(path) as f:
        for line in f:
            current=line.split()
            data=(np.array([float(current[1]),float(current[2]),float(current[3])]).astype(np.float32)-1)/div
            if(math.isnan(data[0])):
                continue
            image_dict[current[0]]=data
    return image_dict

def generate_IAPS_dataset(path,mode):
    image_path=os.path.join(path,"Dataset")
    # return cd.key_dataset(read_txt(os.path.join(path,"IAPS_VAD.txt")),image_path,mode,resize_transforms,
            # target_file_path=["","",""],load_mode=True)
    return cd.dict_dataset(read_txt(os.path.join(path,"IAPS_VAD.txt")),image_path,mode,resize_transforms,load_mode=False)

def generate_NAPS_dataset(path,mode):
    image_path=os.path.join(path,"Dataset")
    # return cd.key_dataset(read_txt(os.path.join(path,"NAPS_VAD.txt")),image_path,mode,resize_transforms,
            # target_file_path=["","",""],load_mode=True)

    return cd.dict_dataset(read_txt(os.path.join(path,"NAPS_VAD.txt")),image_path,mode,resize_transforms,load_mode=False)

def generate_EMOTIC_dataset(path,mode):
    image_path=os.path.join(path,"Dataset")
    train_dict=read_txt(os.path.join(path,"EMOTIC_VAD_train.txt"),9.0)
    test_dict=read_txt(os.path.join(path,"EMOTIC_VAD_test.txt"),9.0)
    val_dict=read_txt(os.path.join(path,"EMOTIC_VAD_val.txt"),9.0)

    total_dict={**train_dict,**test_dict,**val_dict}
    # total_dict={**train_dict,**test_dict,**val_dict}
    # return cd.key_dataset(total_dict,image_path,mode,resize_transforms,
            # target_file_path=["","",""],load_mode=True)

    return cd.dict_dataset(total_dict,image_path,mode,resize_transforms,load_mode=False)

def generate_FI_dataset(path,mode):
    return FI_dataset.FI_dataset(mode,path)

def param_dict_producer(path,dataset,batch_size,epochs):
    if(dataset=="IAPS"):
        generate_function=generate_IAPS_dataset
    if(dataset=="NAPS"):
        generate_function=generate_NAPS_dataset
    if(dataset=="EMOTIC"):
        generate_function=generate_EMOTIC_dataset
    if(dataset=="FI"):
        generate_function=generate_FI_dataset

    param_dict={}
    param_dict["train_loader"]=Data.DataLoader(generate_function(path,"train"),
            batch_size=batch_size,shuffle=False,num_workers=8)
    param_dict["test_loader"]=Data.DataLoader(generate_function(path,"test"),batch_size=2,shuffle=False,num_workers=32)
    param_dict["val_loader"]=Data.DataLoader(generate_function(path,"val"),batch_size=2,shuffle=False,num_workers=32)
    param_dict["epochs"]=epochs
    return param_dict
