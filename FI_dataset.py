import torch
import cv2
import os
from PIL import Image,ImageFile
from torchvision import transforms
import numpy as np
import random

data_transforms = {
        'train': transforms.Compose([
            # transforms.Scale(256),
            transforms.RandomSizedCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.RandomSizedCrop(448),
            # transforms.Scale(600),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            # transforms.RandomSizedCrop(448),
            # transforms.Scale(600),
            transforms.TenCrop(448),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
        ])
}

class FI_dataset(torch.utils.data.Dataset):
    def __init__(self,mode="train",path="./FI"):
        print("reading files")
        self.image_dict={}
        self.mode=mode
        self.transform=None
        if(self.mode=="train"):
            self.transform=data_transforms["train"]
        else:
            self.transform=data_transforms["val"]

        label=0
        for file in os.listdir(path):
            if(file=="Dataset"):
                continue
            current_file=os.path.join(path,file)
            f=open(current_file)
            while True:
                line=f.readline()
                if line:
                    line=line.strip()
                    self.image_dict[line]=label
                else:
                    break
            f.close()
            label+=1

        self.train_data_numbers=int(len(self.image_dict)*0.8)
        self.val_data_numbers=int(len(self.image_dict)*0.05)
        self.test_data_numbers=int(len(self.image_dict))-self.train_data_numbers-self.val_data_numbers
        self.image_dict_list=list(self.image_dict.keys())
        random.seed(666)
        random.shuffle(self.image_dict_list)
        self.path=os.path.join(path,"Dataset")


    def __len__(self):
        if(self.mode=="train"):
            return self.train_data_numbers
        if(self.mode=="val"):
            return self.val_data_numbers
        if(self.mode=="test"):
            return self.test_data_numbers


    def __getitem__(self,index):
        if(self.mode=="train"):
            label=self.image_dict[self.image_dict_list[index]]
            label=torch.LongTensor([label])
            image_name=os.path.join(self.path,self.image_dict_list[index])
            # fp=open(image_name)
            # print(image_name)
            image=Image.open(image_name).convert('RGB')
            image=self.transform(image)
            return image,label[0]

        if(self.mode=="val"):
            # index_yu=index%5
            # index=int(index/5)
            label=self.image_dict[self.image_dict_list[index+self.train_data_numbers]]
            label=torch.LongTensor([label])
            image_name=os.path.join(self.path,self.image_dict_list[index+self.train_data_numbers])
            # fp=open(image_name)
            # print(image_name)
            image=Image.open(image_name).convert('RGB')
            image=self.transform(image)
            # image=val_data_transform(image,index_yu)
            return image,label[0]

        if(self.mode=="test"):
            # index_yu=index%5
            # index=int(index/5)
            label=self.image_dict[self.image_dict_list[index+self.train_data_numbers+self.val_data_numbers]]
            label=torch.LongTensor([label])
            image_name=os.path.join(self.path,self.image_dict_list[index+self.train_data_numbers+self.val_data_numbers])
            # fp=open(image_name)
            image=Image.open(image_name).convert('RGB')
            image=self.transform(image)
            # image=val_data_transform(image,index_yu)
            # fp.close()
            return image,label[0]
