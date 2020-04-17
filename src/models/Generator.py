# -*- coding: utf-8 -*-

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, image_size = 224):
        super(Generator, self).__init__()
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        #Input is of shape(1 X 1 X image_size X image_size)
        self.layer_1 = nn.Conv2d(1,64, 4, padding= 1, stride= 2)
        self.layer_1_bn = nn.BatchNorm2d(64)
        #Output is of shape (1 X 64 X 112 X 112)
        
        self.layer_2 = nn.Conv2d(64,128, 4, padding= 1, stride= 2)
        self.layer_2_bn = nn.BatchNorm2d(128)
        #Output is of shape (1 X 64 X 56 X 56)
        
        self.layer_3 = nn.Conv2d(128,256, 4, padding= 1, stride= 2)
        self.layer_3_bn = nn.BatchNorm2d(256)
        #Output is of shape (1 X 64 X 28 X 28)
        
        self.layer_4 = nn.Conv2d(256,512, 4, padding= 1, stride= 2)
        self.layer_4_bn = nn.BatchNorm2d(512)
        #Output is of shape (1 X 64 X 14 X 14)
        
        self.layer_5 = nn.Conv2d(512,512, 4, padding= 1, stride= 2)
        self.layer_5_bn = nn.BatchNorm2d(512)
        #Output is of shape (1 X 64 X 7 X 7)
        
        self.layer_6 = nn.ConvTranspose2d(512, 512, 4, padding= 1, stride= 2)
        self.layer_6_bn = nn.BatchNorm2d(512)
        #Ouput shape 14 X 14
        
        self.layer_7 = nn.ConvTranspose2d(512, 256, 4, padding= 1, stride= 2)
        self.layer_7_bn = nn.BatchNorm2d(256)
        #Ouput shape 28 X 28
        
        self.layer_8 = nn.ConvTranspose2d(256, 128, 4, padding= 1, stride= 2)
        self.layer_8_bn = nn.BatchNorm2d(128)
        #Output shape 56 X 56
        
        self.layer_9 = nn.ConvTranspose2d(128, 64, 4, padding= 1, stride= 2)
        self.layer_9_bn = nn.BatchNorm2d(64)
        #Ouput shape 112 X 112
        
        self.layer_10 = nn.ConvTranspose2d(64, 3, 4, padding= 1, stride= 2)
        #Output shape is 224 X 224
        
        self._initialize_weights()
    
    def forward(self, x):
        
        x = self.relu(self.layer_1_bn(self.layer_1(x)))
        store_1 = x
        
        x = self.relu(self.layer_2_bn(self.layer_2(x)))
        store_2 = x
        
        x = self.relu(self.layer_3_bn(self.layer_3(x)))
        store_3 = x
        
        x = self.relu(self.layer_4_bn(self.layer_4(x)))
        store_4 = x
        
        x = self.relu(self.layer_5_bn(self.layer_5(x)))
        
        x = self.relu(self.layer_6_bn(self.layer_6(x)))
        x += store_4
        
        x = self.relu(self.layer_7_bn(self.layer_7(x)))
        x += store_3
        
        x = self.relu(self.layer_8_bn(self.layer_8(x)))
        x += store_2
        
        x = self.relu(self.layer_9_bn(self.layer_9(x)))
        x += store_1
        
        x = self.tanh(self.layer_10(x))
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, std= 0.02)
                nn.init.constant_(m.bias.data,0)