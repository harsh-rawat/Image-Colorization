# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
from models.Generator import Generator
from models.Discriminator import Discriminator
import matplotlib.pyplot as plt

class Model:
    def __init__(self, epochs = 10, learning_rate = 0.0002, image_size = 224, leaky_relu = 0.2, betas = (0.5,0.999), lamda = 100):
        self.image_size = image_size
        self.leaky_relu_threshold = leaky_relu
        
        self.epochs = epochs
        self.lr = learning_rate
        self.betas = betas
        self.lamda = lamda
        
        self.gen = None
        self.dis = None
        
        self.gen_optim = None
        self.dis_optim = None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        self.device = None
        
        #Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage -')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
            self.device = device
    
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
    
    def train_model(self, trainloader):
        
        mean_loss = nn.BCELoss()
        l1_loss = nn.L1Loss()
        
        self.gen.train()
        self.dis.train()
        
        gen_loss = []
        dis_loss = []
        
        for i in range(self.epochs):
            running_gen_loss = 0
            running_dis_loss = 0
            
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

                gen_adv_loss = mean_loss(self.dis(fake_img),one_label)
                gen_l1_loss = l1_loss(fake_img.view(batch_size, -1), real_img.view(batch_size, -1))
            
                total_gen_loss = gen_adv_loss + gen_l1_loss
                total_gen_loss.backward()
                self.gen_optim.step()
                    
                running_dis_loss = total_dis_loss.item()
                running_gen_loss = total_gen_loss.item()
                
            gen_loss.append(running_gen_loss)
            dis_loss.append(running_dis_loss)
        
        plt.plot(gen_loss, label = 'Generator Loss')
        plt.plot(dis_loss, label = 'Discriminator Loss')
        plt.legend()
        plt.show()
        
        return (gen_loss, dis_loss)
    
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
    
    def save_checkpoint(self, filename, model_type):
        if(self.gen == None or self.dis == None):
            raise Exception('The model has not been initialized and hence cannot be saved !')
        
        filename = './checkpoints/{}'.format(filename)
        save_dict = {'model_type': model_type, 'dis_dict':self.dis.state_dict(), 'gen_dict': self.gen.state_dict(), 'lr': self.lr,
                    'epochs' : self.epochs, 'betas': self.betas, 'image_size':self.image_size, 
                     'leaky_relu_thresh' : self.leaky_relu_threshold, 'lamda' : self.lamda}
        
        torch.save(save_dict, filename)
        
        print('The model checkpoint has been saved !')
    
    def load_checkpoint(self, filename, model_type):
        print('The model checkpoint has been restored')