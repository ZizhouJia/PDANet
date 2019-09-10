import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os

from torchvision import transforms
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

class extra_branch(nn.Module):
    def __init__(self,in_channels,out_size=14):
        super(extra_branch,self).__init__()
        self.seq=nn.Sequential(nn.Conv2d(in_channels,128,3,1,1),nn.ReLU(True),
                nn.Conv2d(128,128,1,1),nn.ReLU(True),
                nn.Conv2d(128,1,1,1),nn.ReLU(True),
                nn.AdaptiveAvgPool2d(out_size))

    def forward(self,x):
        return self.seq(x)



class fcn_model(nn.Module):
    def __init__(self):
        super(fcn_model,self).__init__()
        self.base_model=torchvision.models.vgg16(pretrained=True)
        self.seq_lists=list(list(self.base_model.children())[0])
        self.seq1=nn.Sequential(*self.seq_lists[0:5])
        self.seq2=nn.Sequential(*self.seq_lists[5:10])
        self.seq3=nn.Sequential(*self.seq_lists[10:17])
        self.seq4=nn.Sequential(*self.seq_lists[17:23])
        self.extra_branch1=extra_branch(64)
        self.extra_branch2=extra_branch(128)
        self.extra_branch3=extra_branch(256)
        self.extra_branch4=extra_branch(512)
        self.final=nn.Conv2d(4,1,1,1)



    def forward(self,x):
        feature=self.seq1(x)
        b1=self.extra_branch1(feature)
        feature=self.seq2(feature)
        b2=self.extra_branch2(feature)
        feature=self.seq3(feature)
        b3=self.extra_branch3(feature)
        feature=self.seq4(feature)
        b4=self.extra_branch4(feature)
        final_map=torch.cat([b1,b2,b3,b4],dim=1)
        out=self.final(final_map)
        out=F.sigmoid(out)
        return out


class msraB(torch.utils.data.Dataset):
    def __init__(self,path="./MSRA-B"):
        self.path=path
        self.file_list=os.listdir(self.path)
        file_list_keys={}
        for name in self.file_list:
            file_list_keys[name[:-4]]=1
        self.file_list=list(file_list_keys.keys())
        self.transform_x=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.transform_y=transforms.Compose([transforms.ToTensor()])
        self.buffer_dict_x={}
        self.buffer_dict_y={}

    def read_image(self,key_name):
        image_name_x=os.path.join(self.path,key_name+".jpg")
        image_name_y=os.path.join(self.path,key_name+".png")
        image_x=Image.open(image_name_x).convert("RGB")
        image_y=Image.open(image_name_y).convert("1")
        return image_x.resize((224,224),Image.BILINEAR),image_y.resize((14,14),Image.BILINEAR)


    
    def __getitem__(self,index):
        key=self.file_list[index]
        if(key in self.buffer_dict_x):
            image_x=self.buffer_dict_x[key]
            image_y=self.buffer_dict_y[key]
            return self.transform_x(image_x),self.transform_y(image_y)
        else:
            image_x,image_y=self.read_image(key)
            self.buffer_dict_x[key]=image_x
            self.buffer_dict_y[key]=image_y
            return self.transform_x(image_x),self.transform_y(image_y)

    def __len__(self):
        return len(self.file_list)

if __name__=="__main__":
    #create dataset
    dataset=msraB()
    dataloader=torch.utils.data.DataLoader(dataset,16,shuffle=True,num_workers=0)

    #create model
    fcn=fcn_model().cuda()

    #create optimizer
    seq_param=fcn.base_model.parameters()
    base_params=list(map(id,seq_param))
    branch_param=filter(lambda p: id(p) not in base_params,fcn.parameters())
    optimizer=torch.optim.SGD([{"params":seq_param,"lr":0.001},
        {"params":branch_param,"lr":0.01}],0.001,weight_decay=0.0005,momentum=0.9)

    loss_function=nn.BCELoss(reduction="mean")


    for epoch in range(0,60):
        print("save model in epoch %d"%(epoch))
        torch.save(fcn.state_dict(),"model.pth")
        for step, data in enumerate(dataloader):
            x,y=data
            x=x.cuda()
            y=y.cuda()
            out=fcn(x)
            out=out.view(out.size(0),-1)
            y=y.view(y.size(0),-1)
            loss=loss_function(out,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if(step%20==0):
                print("epoch %d step %d loss is: %.4f"%(epoch,step,loss.item()))


