from data.Dataloader import Dataloader
from video_colorization.video_utils import *
import torch
import gc

path = 'C:/Users/harsh/Downloads/Assignments/Spring 2020/CS 766 Computer Vision/Project/Data/Animated/Videos'
filename = 'short.mp4'
save_ = 'short_gray.mp4';
global_path = base_path = '{}/{}'.format(path, 'test')
foldername = 'grayscale'

ob = video_utils(path)
#ob.convert_to_grayscale_video(foldername,filename, save_)

loader = Dataloader('{}/{}'.format(path, 'grayscale_gray'), 256, batch_size=100, image_format='jpg',
                    validation_required=(False, 0.2, 'train_validation_split'))
trainloader, _ = loader.get_data_loader()

model = None
torch.cuda.empty_cache()
gc.collect()
#model = Model(base_path = global_path, image_size= 256, image_format = 'jpg')
#average_loss = AverageLoss(global_path)
#model.initialize_model()

ob.apply_ml_model(trainloader, model)

img, _ = next(iter(trainloader))
print(img[99])
