# -*- coding: UTF-8 -*-
import torch
import torch.utils.data as Data


class data_provider:
    def __init__(self, dataset, batch_size, is_cuda=False):
        self.batch_size = batch_size
        self.dataset = dataset
        self.is_cuda = is_cuda  # 是否将batch放到gpu上
        self.dataiter = None
        self.iteration = 0  # 当前epoch的batch数
        self.epoch = 0  # 统计训练了多少个epoch

    def build(self):
        dataloader = Data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.dataiter = iter(dataloader)

    def next(self):
        if self.dataiter is None:
            self.build()
        try:
            batch = self.dataiter.next()
            self.iteration += 1

            if self.is_cuda:
                batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
            return batch

        except StopIteration:  # 一个epoch结束后reload
            self.epoch += 1
            self.build()
            self.iteration = 1  # reset and return the 1st batch

            batch = self.dataiter.next()
            if self.is_cuda:
                batch = [batch[0].cuda(), batch[1].cuda(), batch[2].cuda()]
            return batch
