# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:59:47 2020

@author: harsh
"""

from data.Dataloader import Dataloader
from models.Model import Model

local_path = '/content/data'
local_path_test = '/content/test_data'
global_path = '/content/drive/My Drive/Machine Learning - Google Colab/Flower Dataset/ML_Save_Data'
image_format = 'jpg'
image_size = 256

loader = Dataloader(global_path, image_size, image_format= image_format )
trainloader, validloader = loader.get_data_loader()

test_loader = Dataloader(local_path_test, image_size, image_format= image_format )
test_trainloader, test_validloader = test_loader.get_data_loader()

model = Model(base_path = global_path, image_size= image_size, image_format = image_format)

model.initialize_model()

model.change_params(epochs=55)
model.change_params(learning_rate = 0.0001)
model.change_params(lamda = 100)


model.load_checkpoint('checkpoint_iter_8600')

model.evaluate_model(test_trainloader, 'test', 28)

model.train_model(trainloader, save_model= (True, 100), display_test_image=(True, trainloader, 20))