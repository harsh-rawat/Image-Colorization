# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
from models.Generator import Generator
from models.Discriminator import Discriminator
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import pathlib

class Model:
    def __init__(self, base_path = '', epochs = 10, learning_rate = 0.0002, image_size = 224, leaky_relu = 0.2, betas = (0.5,0.999), lamda = 100, image_format = 'png'):
        self.image_size = image_size
        self.leaky_relu_threshold = leaky_relu
        
        self.epochs = epochs
        self.lr = learning_rate
        self.betas = betas
        self.lamda = lamda
        self.base_path = base_path
        self.image_format = image_format
        self.count = 1
        
        self.gen = None
        self.dis = None
        
        self.gen_optim = None
        self.dis_optim = None
        
        self.device = self.get_device()
    
    def get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        
        #Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage -')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
            return device
        else:
            return None
    
    def initialize_model(self, model_type = 'unet'):
        
        if(model_type == 'unet'):
            self.gen = Generator(image_size= self.image_size)
            self.dis = Discriminator(image_size= self.image_size, leaky_relu= self.leaky_relu_threshold)
            
            self.gen.double()
            self.dis.double()
            
            if(self.device != None):
                self.gen.cuda()
                self.dis.cuda()
            
            self.gen_optim = optim.Adam(self.gen.parameters(), lr= self.lr, betas= self.betas)
            self.dis_optim = optim.Adam(self.dis.parameters(), lr= self.lr, betas= self.betas)
            print('Model Initialized !')
    
    def train_model(self, trainloader, save_model = (False, 25), display_test_image = (False, None, 25)):
        
        mean_loss = nn.BCELoss()
        l1_loss = nn.L1Loss()
        
        self.gen.train()
        self.dis.train()
        
        gen_loss = []
        dis_loss = []
        
        iterations = 1
        
        sample_img_test = None
        if display_test_image[0]:
            gray_test_images, rgb_test_images = next(iter(display_test_image[1]))
            sample_img_test = gray_test_images[0].view(1,1,self.image_size, self.image_size)
            if self.device != None:
                sample_img_test = sample_img_test.cuda()
                save_image((rgb_test_images[0].detach().cpu()+1)/2, '{}/real_img.{}'.format(self.base_path, self.image_format))
        
        running_gen_loss = 0
        running_dis_loss = 0

        for i in range(self.epochs):
            
            for gray_img, real_img in trainloader:
                
                batch_size = len(gray_img)
                zero_label = torch.zeros(batch_size, dtype = torch.double)
                one_label = torch.ones(batch_size, dtype = torch.double)
  
                if self.device != None:
                    gray_img = gray_img.cuda()
                    real_img = real_img.cuda()
                    zero_label = zero_label.cuda()
                    one_label = one_label.cuda()
                    
                #Discriminator loss
                self.dis_optim.zero_grad()
                fake_img = self.gen(gray_img)
                
                dis_real_loss = mean_loss(self.dis(real_img), one_label)
                dis_fake_loss = mean_loss(self.dis(fake_img), zero_label)
                    
                total_dis_loss = dis_fake_loss + dis_real_loss
                total_dis_loss.backward()
                self.dis_optim.step()
                    
                #Generator loss
                self.gen_optim.zero_grad()
                
                fake_img = self.gen(gray_img)
                gen_adv_loss = mean_loss(self.dis(fake_img),one_label)
                gen_l1_loss = l1_loss(fake_img.view(batch_size, -1), real_img.view(batch_size, -1))
            
                total_gen_loss = gen_adv_loss + self.lamda*gen_l1_loss
                total_gen_loss.backward()
                self.gen_optim.step()
                    
                running_dis_loss += total_dis_loss.item()
                running_gen_loss += total_gen_loss.item()                                
                
                if(display_test_image[0] and iterations % display_test_image[2] == 0):
                    self.gen.eval()
                    out_result = self.gen(sample_img_test)
                    out_result = out_result.detach().cpu()
                    out_result = (out_result[0] + 1)/2
                    save_image(out_result,'{}/iteration_{}.{}'.format(self.base_path, iterations, self.image_format))
                    self.gen.train()
                iterations += 1
                
                if(save_model[0] and iterations % save_model[1] == 0):
                    self.save_checkpoint('checkpoint_iter_{}'.format(iterations), 'unet')
                
                if (iterations % 20 == 0):
                    running_gen_loss /= 20.0
                    running_dis_loss /= 20.0
                    print('Losses after iteration : {} : Discriminator Loss : {} and Generator Loss : {}'.format(iterations, running_dis_loss, running_gen_loss))
                    gen_loss.append(running_gen_loss)
                    dis_loss.append(running_dis_loss)
                    running_gen_loss = 0
                    running_dis_loss = 0
        
        plt.plot(gen_loss, label = 'Generator Loss')
        plt.plot(dis_loss, label = 'Discriminator Loss')
        plt.legend()
        plt.show()
        
        return (gen_loss, dis_loss)
    
    def evaluate_model(self, loader, save_filename, no_of_images = 1):
        
        counter_images_generated = 0
        while(counter_images_generated < no_of_images):
          gray, rgb = next(iter(loader))
          test_img = gray[5]
          real_img = rgb[5]
          
          if(self.device != None):
              test_img = test_img.cuda()
          
          if(self.gen == None or self.dis == None):
              raise Exception('Model has not been initialized!')
          
          filename = '{}/{}_{}.{}'.format(self.base_path, save_filename, self.count, self.image_format)
          real_filename = '{}/{}_{}_real.{}'.format(self.base_path, save_filename, self.count, self.image_format)
          real_gray_filename = '{}/{}_{}_real_gray.{}'.format(self.base_path, save_filename, self.count, self.image_format)
          self.count += 1
          
          self.gen.eval()
          test_img = test_img.view(1,1,self.image_size, self.image_size)
          out = self.gen(test_img)
          out = out.detach().cpu()
          out = (out+1)/2
          save_image(out, filename)
          
          gray_img = gray[5].detach().cpu()
          save_image(gray_img, real_gray_filename)

          real_img = (real_img.detach().cpu() +1)/2
          save_image(real_img, real_filename)

          counter_images_generated +=1
    
    def change_params(self, epochs = None, learning_rate = None, leaky_relu = None, betas = None, lamda = None):
        if(epochs != None):
            self.epochs = epochs
            print('Changed the number of epochs!')
        if(learning_rate != None):
            self.lr = learning_rate
            print('Changed the learning rate!')
        if(leaky_relu != None):
            self.leaky_relu_threshold = leaky_relu
            print('Changed the threshold for leaky relu!')
        if(betas != None):
            self.betas = betas
            print('Changed the betas for Adams Optimizer!')
        if(betas != None or learning_rate != None):
            self.gen_optim = optim.Adam(self.gen.parameters(), lr= self.lr, betas= self.betas)
            self.dis_optim = optim.Adam(self.dis.parameters(), lr= self.lr, betas= self.betas)
            
        if(lamda != None):
            self.lamda = lamda
            print('Lamda value has been changed!')
    
    def save_checkpoint(self, filename, model_type = 'unet'):
        if(self.gen == None or self.dis == None):
            raise Exception('The model has not been initialized and hence cannot be saved !')
        
        filename = '{}/checkpoints/{}.pth'.format(self.base_path, filename)
        save_dict = {'model_type': model_type, 'dis_dict':self.dis.state_dict(), 'gen_dict': self.gen.state_dict(), 'lr': self.lr,
                    'epochs' : self.epochs, 'betas': self.betas, 'image_size':self.image_size, 
                     'leaky_relu_thresh' : self.leaky_relu_threshold, 'lamda' : self.lamda, 'base_path': self.base_path, 
                     'count' : self.count, 'image_format': self.image_format}
        
        torch.save(save_dict, filename)
        
        print('The model checkpoint has been saved !')
    
    def load_checkpoint(self, filename):
        filename = '{}/checkpoints/{}.pth'.format(self.base_path, filename)
        if(not pathlib.Path(filename).exists()):
            raise Exception('This checkpoint does not exist!')
        
        self.gen = None
        self.dis = None
        
        save_dict = torch.load(filename)
        
        self.betas = save_dict['betas']
        self.image_size = save_dict['image_size']
        self.epochs = save_dict['epochs']
        self.leaky_relu_threshold = save_dict['leaky_relu_thresh']
        self.lamda = save_dict['lamda']
        self.lr = save_dict['lr']
        self.base_path = save_dict['base_path']
        self.count = save_dict['count']
        self.image_format = save_dict['image_format']
        self.device = self.get_device()
        
        self.initialize_model(model_type= save_dict['model_type'])
        
        self.gen.load_state_dict(save_dict['gen_dict'])
        self.dis.load_state_dict(save_dict['dis_dict'])
        
        print('The model checkpoint has been restored')