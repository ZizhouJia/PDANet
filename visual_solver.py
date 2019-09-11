import os
import cv2
import numpy as np
import model_utils.solver as solver
from skimage.transform import resize

class visual_solver(solver.solver):
    def __init__(self, models,optimizers,kernel_processer,model_name,save_path,restore_time_string):
        super(visual_solver, self).__init__(
            models,optimizers,kernel_processer,model_name,save_path)
        self.time_string=restore_time_string
        self.images=[]
        self.real_labels=[]
        self.pred_labels=[]

        self.std=np.array([[[0.229, 0.224, 0.225]]])
        self.mean=np.array([[[0.485, 0.456, 0.406]]])

    def test_model(self, param_dict,mode="test"):
        pass

    def evaluate_model(self,param_dict,mode="val"):
        pass


    def train_model(self,epoch,param_dict):
        pass

    def write_file(self):
        path_name="visual_ours"
        if(not os.path.exists(path_name)):
            os.makedirs(path_name)
        path_name=os.path.join(path_name,self.model_name)
        if(not os.path.exists(path_name)):
            os.makedirs(path_name)
        for i in range(0,len(self.images)):
            cv2.imwrite(os.path.join(path_name,str(i)+".jpg"),self.images[i])
        txt_file="real_label.txt"
        t=open(os.path.join(path_name,txt_file),"w")
        for i in range(0,len(self.real_labels)):
            t.write("   "+str(self.real_labels[i][0])+" "+str(self.real_labels[i][1])+" "+str(self.real_labels[i][2])+"\n")
        t.close()

    def main(self,param_dict):
        self.restore_params(self.time_string,"best")
        epochs=param_dict["epochs"]
        dataloader=param_dict["test_loader"]
        self.eval_mode()
        for step,data in enumerate(dataloader):
            print("step: "+str(step))
            for i in range(0,len(data)):
                data[i]=data[i].cuda()
            reals,preds,images,attention_maps=self.kernel_processer.evaluate(step,data)
            for i in range(0,reals.shape[0]):
                self.real_labels.append(reals[i,:])
                self.pred_labels.append(preds[i,:])
                c_image=images[i,:,:,:].transpose(1,2,0)
                c_attention_map=attention_maps[i,:,:,:].transpose(1,2,0)
                at_min=c_attention_map.min()
                at_max=c_attention_map.max()
                c_attention_map=(c_attention_map-at_min)/(at_max-at_min)*0.8
                # c_attention_map=np.repeat(c_attention_map,3,axis=2)
                c_attention_map=resize(c_attention_map[:,:,0],(448,448),preserve_range=True)
                # c_attention_map=cv2.blur(c_attention_map,(50,50))
                c_attention_map=(c_attention_map*255).astype(np.uint8)
                cam_heatmap=cv2.applyColorMap(c_attention_map,cv2.COLORMAP_JET)
                # cam_heatmap=cv2.cvtColor(cam_heatmap,cv2.COLOR_BGR2RGB)
                c_image=cv2.resize(c_image*self.std+self.mean,(448,448))
                c_image=(c_image*255).astype(np.uint8)
                c_image=cv2.cvtColor(c_image,cv2.COLOR_RGB2BGR)
                image_combine=cv2.addWeighted(c_image,0.5,cam_heatmap,0.5,0.0)
                final_image=np.concatenate((c_image,cam_heatmap,image_combine),axis=1)
                self.images.append(final_image)
        self.write_file()
        return 1,1
