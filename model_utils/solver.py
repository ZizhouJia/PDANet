import os
from datetime import datetime
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torchvision.utils as vutils


def get_time_string():
    dt = datetime.now()
    return dt.strftime("%Y%m%d%H%M")


class solver(object):
    def __init__(self, models,optimizers,kernel_processer,model_name,save_path="checkpoints"):
        self.models = models
        self.model_name = model_name
        self.save_path = save_path
        self.time_string = get_time_string()
        self.writer = SummaryWriter("runs/"+self.model_name+"-"+self.time_string)
        self.optimizers = optimizers
        self.kernel_processer=kernel_processer
        self.kernel_processer.set_models(models)
        self.kernel_processer.set_optimizers(optimizers)

    def get_models(self):
        return self.models

    def init_models(self, init_func):
        for model in self.models:
            init_func(model)

    def set_optimizers(self, optimizers):
        self.optimizers = optimizers
        self.kernel_processer.set_optimizers(optimizers)

    def train_mode(self):
        for model in self.models:
            model.train()

    def eval_mode(self):
        for model in self.models:
            model.eval()

    def write_log(self, loss, index):
        for key in loss:
            self.writer.add_scalar("scalar/"+key, loss[key], index)

    def output_loss(self, loss, epoch, iteration):
        print("in epoch %d iteration %d " % (epoch, iteration))
        print(loss)

    def write_log_image(self, image, index):
        for key in image:
            self.writer.add_image(
                'image/'+key, vutils.make_grid(image[key], 1), index)

    def save_params(self, epoch=-1):
        path = self.save_path
        if(not os.path.exists(path)):
            os.mkdir(path)

        path = os.path.join(path, self.model_name)
        if(not os.path.exists(path)):
            os.mkdir(path)

        path = os.path.join(path, self.time_string)
        if(not os.path.exists(path)):
            os.mkdir(path)

        if(epoch != -1):
            path = os.path.join(path, str(epoch))
            if(not os.path.exists(path)):
                os.mkdir(path)

        file_name = "model"
        for i in range(0, len(self.models)):
            torch.save(self.models[i].state_dict(), os.path.join(
                path, file_name+"-"+str(i)+".pkl"))

        print("the models params "+self.model_name +
              " has already been saved "+self.time_string)

    def restore_params(self, time_string, epoch=-1):
        path = self.save_path
        path = os.path.join(path, self.model_name)
        path = os.path.join(path, time_string)
        if(epoch != -1):
            path = os.path.join(path, str(epoch))
        file_name = "model"

        for i in range(0, len(self.models)):
            self.models[i].load_state_dict(torch.load(
                os.path.join(path, file_name+"-"+str(i)+".pkl")))

        print("the models params "+self.model_name +
              " has already been restored "+self.time_string)

    def save_models(self, epoch=-1):
        path = self.save_path
        if(not os.path.exists(path)):
            os.mkdir(path)

        path = os.path.join(path, self.model_name)
        if(not os.path.exists(path)):
            os.mkdir(path)

        path = os.path.join(path, self.time_string)
        if(not os.path.exists(path)):
            os.mkdir(path)

        if(epoch != -1):
            path = os.path.join(path, str(epoch))
            if(not os.path.exists(path)):
                os.mkdir(path)

        file_name = "model"
        for i in range(0, len(self.models)):
            torch.save(self.models[i], os.path.join(
                path, file_name+"-"+str(i)+".pkl"))

        print("the models "+self.model_name +
              " has already been saved "+self.time_string)

    def restore_models(self, time_string, epoch=-1):
        path = self.save_path
        path = os.path.join(path, self.model_name)
        path = os.path.join(path, time_string)
        if(epoch != -1):
            path = os.path.join(path, epoch)
        file_name = "model"

        self.models = []
        i = 0
        while(True):
            current_file = os.path.join(path, file_name+"-"+str(i)+".pkl")
            if(not os.path.exists(current_file)):
                break
            self.models.append(torch.load(current_file))
            i += 1

        print("the models params "+self.model_name + \
              " has already been restored "+self.time_string)


    def test_model(self, param_dict):
        raise NotImplementedError


    def train_model(self,epoch,param_dict):
        raise NotImplementedError
