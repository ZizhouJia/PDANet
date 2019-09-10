import torch
import cv2
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import random

data_transforms = {
        'train': transforms.Compose([
            transforms.Scale(600),
            transforms.RandomSizedCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(600),
            transforms.TenCrop(448),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}


class IASP_dataset(torch.utils.data.Dataset):
    def __init__(self,mode="train",path="./IAPSdimension"):
        print("reading files")
        self.image_dict={}
        self.mode=mode
        self.transform=None
        if(self.mode=="train"):
            self.transform=data_transforms["train"]
        else:
            self.transform=data_transforms["val"]

        with open(os.path.join(path,"IAPS_VAD.txt")) as f:
            for line in f:
                current=line.split()
                data=[float(current[1]),float(current[2]),float(current[3])]
                self.image_dict[current[0]]=data
        self.train_data_numbers=int(len(self.image_dict)*0.8)
        self.val_data_numbers=int(len(self.image_dict)*0.1)
        self.test_data_numbers=int(len(self.image_dict))-self.train_data_numbers-self.val_data_numbers
        self.image_dict_list=self.image_dict.keys()
        random.seed(666)
        self.image_dict_list=random.shuffle(self.image_dict_list)
        self.path=os.path.join(path,"Dataset")



    def __len__(self):
        if(self.mode=="train"):
            return self.train_data_numbers
        else:
            return len(self.image_dict)-self.train_data_numbers

    def __getitem__(self,index):
        if(self.mode=="train"):
            label=self.image_dict[self.image_dict_list[index]]
            label=torch.Tensor(np.array(label).astype(np.float32))
            image_name=os.path.join(self.path,self.image_dict_list[index]+".jpg")
            fp=open(image_name)
            image=Image.open(fp)
            image=self.transform(image)
            fp.close()
            return image,label

        if(self.mode=="val")
            label=self.image_dict[self.image_dict_list[index+self.train_data_numbers]]
            label=torch.Tensor(np.array(label).astype(np.float32))
            image_name=os.path.join(self.path,self.image_dict_list[index+self.train_data_numbers]+".jpg")
            fp=open(image_name)
            image=Image.open(fp)
            image=self.transform(image)
            fp.close()
            return image,label

        if(self.mode=="test"):
            label=self.image_dict[self.image_dict_list[index+self.train_data_numbers+self.val_data_numbers]]
            label=torch.Tensor(np.array(label).astype(np.float32))
            image_name=os.path.join(self.path,self.image_dict_list[index+self.train_data_numbers+self.val_data_numbers]+".jpg")
            fp=open(image_name)
            image=Image.open(fp)
            image=self.transform(image)
            fp.close()
            return image,label
