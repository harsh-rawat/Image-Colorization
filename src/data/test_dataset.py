from src.data.Dataloader import *
import numpy as np
import matplotlib.pyplot as plt

data_path = 'C:/Users/harsh/Downloads/Assignments/Spring 2020/CS 766 Computer Vision/Project/Data/17flowers'
image_size = 256
batch_size = 2
image_format = 'jpg'

loader = Dataloader(data_path, image_size, batch_size=batch_size,
                    image_format=image_format, validation_required=(True, 0.2, 'train_validation_split'))
trainloader, validloader = loader.get_data_loader()

generate_sample(trainloader)