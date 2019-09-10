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

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.seq1=nn.Sequential(nn.Conv2d(3,64,kernel_size=3,padding=1),
                nn.ReLU(True),
                nn.Conv2d(64,64,kernel_size=3,padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2,stride=2))
        self.seq2=nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),
                nn.ReLU(True),
                nn.Conv2d(128,128,kernel_size=3,padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2,stride=2))
        self.seq3=nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),
                nn.ReLU(True),
                nn.Conv2d(256,256,kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(256,256,kernel_size=3,padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2,stride=2))
        self.seq4=nn.Sequential(nn.Conv2d(256,512,kernel_size=3,stride=1),
                nn.ReLU(True),
                nn.Conv2d(512,512,kernel_size=3,stride=1),
                nn.ReLU(True),
                nn.Conv2d(512,512,kernel_size=3,stride=1),
                nn.ReLU(True))
        self.seq5=nn.Sequential(nn.Conv2d(512,512,kernel_size=3,stride=1),
                nn.ReLU(True),
                nn.Conv2d(512,512,kernel_size=3,stride=1),
                nn.ReLU(True),
                nn.Conv2d(512,512,kernel_size=3,stride=1),
                nn.ReLU())





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

class WSCNet(nn.Module):
    def __init__(self,k,C):
        super(WSCNet,self).__init__()
        self.base=models.resnet101(pretrained=True)
        self.k=k
        self.C=C
        self.det=detection_branch(self.k,self.C)
        self.classify=classify_branch(self.C)
        self.fc=nn.Linear(2048*2,C)

    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        v,M=self.det(x)
        out=self.classify(x,M)
        out=self.fc(out)
        return out,v,M

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


class senti_block(nn.Module):
    def __init__(self):
        super(senti_block,self).__init__()
        self.tanh=nn.Tanh()
        self.conv1=nn.Conv2d(2048,2048,1)
        self.conv2=nn.Conv2d(2048,1,1)

    def forward(self,x):
        out=self.conv1(x)
        out=self.tanh(out)
        out=self.conv2(out)
        x_shape=out.size(2)
        y_shape=out.size(3)
        out=out.view(-1,x_shape*y_shape)
        out=F.softmax(out,dim=1)
        out=out.view(-1,1,x_shape,y_shape)
        out=out*x
        out=torch.sum(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)
        return out

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


class BasicBlock(nn.Module):
    def __init__(self, num_outputs=[96, 96, 96], num_inputs=3, k_size=1, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, num_outputs[0], k_size, stride)
        self.conv2 = nn.Conv2d(num_outputs[0], num_outputs[1], 1, 1)
        self.conv3 = nn.Conv2d(num_outputs[1], num_outputs[2], 1, 1)
        self.pool = nn.MaxPool2d(3, 2)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class EmoNet(nn.Module):
    def __init__(self):
        super(EmoNet, self).__init__()
        self.block1 = BasicBlock([96, 96, 96], 3, 11, 4)
        self.block2 = BasicBlock([256, 256, 256], 96, 5, 1)
        self.block3 = BasicBlock([384, 384, 384], 256, 5, 1)
        self.block4 = BasicBlock([1024, 512, 256], 384, 4, 1)

        self.side1 = nn.Conv2d(96, 256, 1)
        self.side2 = nn.Conv2d(256, 256, 1)
        self.side3 = nn.Conv2d(384, 256, 1)
        self.side4 = nn.Conv2d(256, 256, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc = nn.Linear(512, 3)


    def forward(self, input):
        branch1 = self.block1(input)
        branch2 = self.block2(branch1)
        branch3 = self.block3(branch2)
        branch4 = self.block4(branch3)

        side1 = self.side1(branch1)
        side1 = F.relu(side1)
        side2 = self.side2(branch2)
        side2 = F.relu(side2)
        side3 = self.side3(branch3)
        side3 = F.relu(side3)
        side4 = self.side4(branch4)
        side4 = F.relu(side4)

        side1 = F.avg_pool2d(side1, side1.size(-1))
        side1 = F.dropout(side1,p=0.5,training=self.training)
        side2 = F.avg_pool2d(side2, side2.size(-1))
        side2 = F.dropout(side2,p=0.5,training=self.training)
        side3 = F.avg_pool2d(side3, side3.size(-1))
        side3 = F.dropout(side3,p=0.5,training=self.training)
        side4 = F.avg_pool2d(side4, side4.size(-1))
        side4 = F.dropout(side4,p=0.5,training=self.training)

        feat = torch.cat([side1, side2, side3, side4], dim=2)
        feat = F.relu(feat)
        feat = self.fc1(feat.view(feat.size(0), -1))
        feat = F.relu(feat)
        feat = self.fc(feat)

        # feat = torch.sigmoid(feat)
        return feat,feat,feat

class view_layer(nn.Module):
    def __init__(self):
        super(view_layer,self).__init__()

    def forward(self,x):
        x=x.view(x.size(0),-1)
        return x

def _upsample_add(x,y):
    _,_,H,W=y.size()
    return F.upsample(x,size=(H,W),mode='bilinear')+y

def _global_avg_pool(x):
    x=torch.mean(x.view(x.size(0),x.size(1),-1),2)
    return x

class fc_layer(nn.Module):
    def __init__(self,in_feature=128,out_feature=1024):
        super(fc_layer,self).__init__()
        self.model=nn.Sequential(nn.Linear(in_feature,out_feature),
                nn.ReLU(True),
                nn.Linear(out_feature,out_feature),
                nn.ReLU(True))

    def forward(self,x):
        return self.model(x)


class mldr_net(nn.Module):
    def __init__(self):
        super(mldr_net,self).__init__()
        self.block1=nn.Sequential(nn.Conv2d(3,128,kernel_size=11,stride=1,padding=5),
                nn.ReLU(True),
                nn.Conv2d(128,128,1,1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3,stride=2))

        self.block2=nn.Sequential(nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2),
                nn.ReLU(True),
                nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3,stride=2))

        self.block3=nn.Sequential(nn.Conv2d(256,384,kernel_size=5,stride=1,padding=2),
                nn.ReLU(True),
                nn.Conv2d(384,384,1,1,0),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3,stride=2))

        self.block4=nn.Sequential(nn.Conv2d(384,512,kernel_size=5,stride=1,padding=2),
                nn.ReLU(True),
                nn.Conv2d (512,512,1,1,0),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3,stride=2))

        self.conv4_1_1=nn.Conv2d(512,128,1)
        self.conv3_1_1=nn.Conv2d(384,128,1)
        self.conv2_1_1=nn.Conv2d(256,128,1)
        self.conv1_1_1=nn.Conv2d(128,128,1)

        self.fc1=fc_layer()
        self.fc2=fc_layer()
        self.fc3=fc_layer()
        self.fc4=fc_layer()
        
        self.fc=nn.Linear(1024,3)
        self.apply(self.weight_init)

    def weight_init(self,m):
        class_name=m.__class__.__name__
        if(class_name.find("Conv")!=-1):
            torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.zeros_(m.bias.data)


    def forward(self,x):
        feature1=self.block1(x)
        feature2=self.block2(feature1)
        feature3=self.block3(feature2)
        feature4=self.block4(feature3)
        fm4=F.relu(self.conv4_1_1(feature4))
        fm3=F.relu(_upsample_add(fm4,self.conv3_1_1(feature3)))
        fm2=F.relu(_upsample_add(fm3,self.conv2_1_1(feature2)))
        fm1=F.relu(_upsample_add(fm2,self.conv1_1_1(feature1)))
        fm4=_global_avg_pool(fm4)
        fm3=_global_avg_pool(fm3)
        fm2=_global_avg_pool(fm2)
        fm1=_global_avg_pool(fm1)
        out1=self.fc1(fm1)
        out2=self.fc2(fm2)
        out3=self.fc3(fm3)
        out4=self.fc4(fm4)
        out=out1+out2+out3+out4
        out=out/4
        out=self.fc(out)
        return out,out,out

