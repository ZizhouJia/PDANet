#change the newwork structure
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class resnet_base(nn.Module):
    def __init__(self):
        super(resnet_base,self).__init__()
        self.base=models.resnet101(pretrained=True)

    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        out=x
        return torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)

class alexnet_base(nn.Module):
    def __init__(self):
        super(alexnet_base,self).__init__()
        self.base=models.alexnet(pretrained=True)
        self.classifier=nn.Sequential(*list(self.base.classifier.children())[:-1])

    def forward(self,x):
        out=self.base.features(x)
        out=out.view(-1,out.size(1)*out.size(2)*out.size(3))
        out=self.classifier(out)
        return out

class vgg16_base(nn.Module):
    def __init__(self):
        super(vgg16_base,self).__init__()
        self.base=models.vgg16(pretrained=True)
        self.classifier=nn.Sequential(*list(self.base.classifier.children())[:-2])

    def forward(self,x):
        x=self.base.features(x)
        x=x.view(-1,x.size(1)*x.size(2)*x.size(3))
        x=self.classifier(x)
        return x


class vgg_net(nn.Module):
    def __init__(self,C):
        super(vgg_net,self).__init__()
        self.base=vgg16_base()
        self.fc=nn.Linear(4096,C)

    def forward(self,x):
        out=self.base(x)
        out=self.fc(out)
        return out,out,out

class alex_net(nn.Module):
    def __init__(self,C):
        super(alex_net,self).__init__()
        self.base=alexnet_base()
        self.fc=nn.Linear(4096,C)

    def forward(self,x):
        out=self.base(x)
        out=self.fc(out)
        return out,out,out


class detection_branch(nn.Module):
    def __init__(self,k,C):
        super(detection_branch,self).__init__()
        self.k=k
        self.C=C
        self.conv1=nn.Conv2d(2048,k*C,1)

    def forward(self,x):
        out=self.conv1(x)
        mean_feature=torch.max(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)[0]
        v=torch.mean(mean_feature.view(-1,self.C,self.k),2)  #v is [-1,C]
        feature_map=torch.mean(out.view(-1,self.C,self.k,out.size(2),out.size(3)),2)
        M=torch.sum((v.view(-1,self.C,1,1)*feature_map),1) #M is [-1,1,H,W]
        return v,M.view(M.size(0),1,M.size(1),M.size(2))


class classify_branch(nn.Module):
    def __init__(self,C):
        super(classify_branch,self).__init__()
    def forward(self,x,M):
        out=torch.cat((x,x*M),1)
        out=torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)
        return out


class Baseline(nn.Module):
    def __init__(self,C):
        super(Baseline,self).__init__()
        self.base=models.resnet101(pretrained=True)
        self.fc=nn.Linear(2048,3)

    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        out=x
        out=torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)
        out=self.fc(out)
        return out,out,out

class SENet_block(nn.Module):
    def __init__(self,activate_type="none"):
        super(SENet_block,self).__init__()
        self.conv1=nn.Conv2d(2048,2048,1)
        self.conv2=nn.Conv2d(2048,2048,1)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.conv_classify=nn.Conv2d(2048,2048,1)
        self.activate_type=activate_type

    def forward(self,x):
        out1=self.conv1(x)
        out1=self.sigmoid(out1)
        out1=torch.mean(out1.view(-1,out1.size(1),out1.size(2)*out1.size(3)),2)
        out_channel_wise=None
        if(self.activate_type=="none"):
            out_channel_wise=out1
        if(self.activate_type=="softmax"):
            out_channel_wise=F.softmax(out1,dim=1)
        if(self.activate_type=="sigmoid"):
            out_channel_wise=F.sigmoid(out1)
        if(self.activate_type=="sigmoid_res"):
            out_channel_wise=F.sigmoid(out1)+1
        out2=self.conv2(x)
        out2=self.relu(out2)
        out=out2*out_channel_wise.view(-1,out1.size(1),1,1)
        return out,out_channel_wise,out2


class spatial_block(nn.Module):
    def __init__(self):
        super(spatial_block,self).__init__()
        self.tanh=nn.Tanh()
        self.fc=nn.Linear(2048,2048)
        self.conv=nn.Conv2d(2048,2048,1)
        self.conv2=nn.Conv2d(2048,1,1)

    def forward(self,x,channel_wise):
        out=self.conv(x)
        if(len(channel_wise.size())!=len(x.size())):
            channel_wise=self.fc(channel_wise)
            channel_wise=channel_wise.view(-1,channel_wise.size(1),1,1)
            out=out+channel_wise
        out=self.tanh(out)
        out=self.conv2(out)
        x_shape=out.size(2)
        y_shape=out.size(3)
        out=out.view(-1,x_shape*y_shape)
        out=F.softmax(out,dim=1)
        out=out.view(-1,1,x_shape,y_shape)
        out=x*out
        out=torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)
        return out



class SENet_base(nn.Module):
    def __init__(self,k,C,type=1):
        super(SENet_base,self).__init__()
        self.base=models.resnet101(pretrained=True)
        self.se_block=None
        self.se_block=SENet_block()
        self.fc=nn.Linear(2048,3)

    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        out,out1,out_classify=self.se_block(x)
        out=torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)
        out=self.fc(out)
        return out,out,out

class SENet_attention_wise(nn.Module):
    def __init__(self,C,activate_type="none"):
        super(SENet_attention_wise,self).__init__()
        self.base=models.resnet101(pretrained=True)
        self.spatial=spatial_block()
        self.fc=nn.Linear(2048,3)
        self.conv1=nn.Conv2d(2048,2048,1)

    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        x=self.conv1(x)
        out=self.spatial(x,x)
        out=self.fc(out)
        return out,out,out

class SENet_senti_attention_wise(nn.Module):
    def __init__(self,C):
        super(SENet_senti_attention_wise,self).__init__()
        self.base=models.resnet101(pretrained=True)
        self.spatial=senti_block()
        self.fc=nn.Linear(2048,3)

    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        out=self.spatial(x)
        out=self.fc(out)
        return out,out,out

class SENet_channel_wise(nn.Module):
    def __init__(self,C,activate_type="none"):
        super(SENet_channel_wise,self).__init__()
        self.base=models.resnet101(pretrained=True)
        self.se_block=SENet_block(activate_type)
        self.fc=nn.Linear(2048,3)
        self.fc_classify=nn.Linear(2048,C)

    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        out,out_channel_wise,out2=self.se_block(x)
        out_mean=torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)
        out=self.fc(out_mean)
        out_classify=self.fc_classify(out_mean)
        return out,out_classify,out


class SENet_channel_wise_with_attention(nn.Module):
    def __init__(self,C,activate_type="none"):
        super(SENet_channel_wise_with_attention,self).__init__()
        self.base=models.resnet101(pretrained=True)
        self.se_block=None
        self.se_block=SENet_block(activate_type)
        self.fc=nn.Linear(2048*2,3)
        self.fc_classify=nn.Linear(2048*2,C)
        self.spatial=spatial_block()

    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        out,out_channel_wise,out2=self.se_block(x)
        out=torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)
        spatial_feature=self.spatial(out2,out_channel_wise)
        feature_cat=torch.cat((out,spatial_feature),1)
        out=self.fc(feature_cat)
        out_classify=self.fc_classify(feature_cat)
        return out,out_classify,out

class SENet(nn.Module):
    def __init__(self,k,C,activate_type="none"):
        super(SENet,self).__init__()
        base_features=2048
        self.base=models.resnet101(pretrained=True)
        self.det=detection_branch(k,C)
        self.se_block=SENet_block(activate_type)
        self.fc=nn.Linear(base_features,C)

    def forward(self,x):
        x=self.base(x)
        v,M=self.det(x)
        out,out1=self.se_block(x)
        out=torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)
        out=self.fc(out)


class SENet_super(nn.Module):
    def __init__(self,k,C,activate_type="none"):
        super(SENet_super,self).__init__()
        self.base=models.resnet101(pretrained=True)
        self.det=detection_branch(k,C)
        self.se_block=SENet_block(activate_type)
        self.fc=nn.Linear(4096,C)

    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        v,M=self.det(x)
        out,out1=self.se_block(x)
        out=torch.cat((out,x*M),1)
        out=torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)
        out=self.fc(out)
        return out,v,M
