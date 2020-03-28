# -*- coding: utf-8 -*-

import torch
from data_module.CustomDataset import CustomDataset
from random import shuffle

class Dataloader:
    def __init__(self, path, image_size, batch_size = 16, image_format = 'png', validation_required = (False, 0.2)):
        self.path = path
        self.image_size = image_size
        self.image_format = image_format
        self.validation_req = validation_required
        self.batch_size = batch_size
    
    def get_data_loader(self):
        dataset = CustomDataset(self.path, self.image_size, self.image_format)
        size = len(dataset)
        
        train_index = list(range(size))
        valid_index = []
        
        if self.validation_req[0]:
            train_size = int(size*(1- self.validation_req[1]))
            indexes = list(range(size))
            shuffle(indexes)
            train_index = indexes[0:train_size]
            valid_index = indexes[train_size:]
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_index)
        
        trainloader = None
        validloader = None
        if (len(valid_index) == 0):
            trainloader = torch.utils.data.DataLoader(dataset, shuffle= True, batch_size= self.batch_size)
        else:
            trainloader = torch.utils.data.DataLoader(dataset, sampler= train_sampler, batch_size= self.batch_size)
            validloader = torch.utils.data.DataLoader(dataset, sampler= valid_sampler, batch_size= self.batch_size)
        
        return (trainloader, validloader)