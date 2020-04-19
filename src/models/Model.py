import pathlib
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from src.models.Discriminator import Discriminator
from src.models.Generator_RESNET import Generator_RESNET
from src.models.Generator_Unet import Generator_Unet


class Model:
    def __init__(self, base_path='', epochs=10, learning_rate=0.0002, image_size=256, leaky_relu=0.2,
                 betas=(0.5, 0.999), lamda=100, image_format='png'):
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
        self.model_type = None

        self.device = self.get_device()
        self.create_folder_structure()

    def create_folder_structure(self):
        checkpoint_folder = self.base_path + '/checkpoints'
        loss_folder = self.base_path + '/Loss_Checkpoints'
        training_folder = self.base_path + '/Training Images'
        test_folder = self.base_path + '/Test Images'
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        if not os.path.exists(loss_folder):
            os.makedirs(loss_folder)
        if not os.path.exists(training_folder):
            os.makedirs(training_folder)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

    def get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print(torch.cuda.get_device_name(0))

        if device.type == 'cuda':
            print('Memory Usage -')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
            return device
        else:
            return None

    def initialize_model(self, model_type='unet'):

        all_models = ['unet', 'resnet']
        if model_type not in all_models:
            raise Exception('This model type is not available!');

        if model_type == 'unet':
            self.gen = Generator_Unet(image_size=self.image_size)
            self.dis = Discriminator(image_size=self.image_size, leaky_relu=self.leaky_relu_threshold)
        elif model_type == 'resnet':
            self.gen = Generator_RESNET()
            self.dis = Discriminator(image_size=self.image_size, leaky_relu=self.leaky_relu_threshold)

        if self.device is not None:
            self.gen.cuda()
            self.dis.cuda()

        self.gen_optim = optim.Adam(self.gen.parameters(), lr=self.lr, betas=self.betas)
        self.dis_optim = optim.Adam(self.dis.parameters(), lr=self.lr, betas=self.betas)

        self.model_type = model_type
        print('Model Initialized !\nGenerator Model Type : {}'.format(model_type))

    def train_model(self, trainloader, average_loss, eval=(False, None, None), save_model=(False, 25),
                    display_test_image=(False, None, 25), change_lr=(False, 30)):

        mean_loss = nn.BCELoss()
        l1_loss = nn.L1Loss()

        self.gen.train()
        self.dis.train()

        iterations = 1
        batches = len(trainloader)
        print('Total number of batches in an epoch are : {}'.format(batches))

        sample_img_test = None
        if display_test_image[0]:
            sample_img_test, rgb_test_images = next(iter(display_test_image[1]))
            save_image((rgb_test_images[0].detach().cpu() + 1) / 2,
                       '{}/Training Images/real_img.{}'.format(self.base_path, self.image_format))
            if self.device is not None:
                sample_img_test = sample_img_test.cuda()

        for i in range(self.epochs):

            if eval[0] and (i % eval[2] == 0):
                self.evaluate_L1_loss_dataset(eval[1], train=False)
                self.gen.train()

            running_gen_loss = 0
            running_dis_loss = 0

            if change_lr[0] and (i + 2) % change_lr[1] == 0:
                self.change_params(learning_rate=self.lr * 0.8)

            for gray_img, real_img in trainloader:

                batch_size = len(gray_img)
                zero_label = torch.zeros(batch_size)
                one_label = torch.ones(batch_size)

                if self.device is not None:
                    gray_img = gray_img.cuda()
                    real_img = real_img.cuda()
                    zero_label = zero_label.cuda()
                    one_label = one_label.cuda()

                # Discriminator loss
                self.dis_optim.zero_grad()
                fake_img = self.gen(gray_img)

                dis_real_loss = mean_loss(self.dis(real_img), one_label)
                dis_fake_loss = mean_loss(self.dis(fake_img), zero_label)

                total_dis_loss = dis_fake_loss + dis_real_loss
                total_dis_loss.backward()
                self.dis_optim.step()

                # Generator loss
                self.gen_optim.zero_grad()

                fake_img = self.gen(gray_img)
                gen_adv_loss = mean_loss(self.dis(fake_img), one_label)
                gen_l1_loss = l1_loss(fake_img.view(batch_size, -1), real_img.view(batch_size, -1))

                total_gen_loss = gen_adv_loss + self.lamda * gen_l1_loss
                total_gen_loss.backward()
                self.gen_optim.step()

                running_dis_loss += total_dis_loss.item()
                running_gen_loss += total_gen_loss.item()

                if display_test_image[0] and iterations % display_test_image[2] == 0:
                    self.gen.eval()
                    out_result = self.gen(sample_img_test)
                    out_result = out_result.detach().cpu()
                    out_result = (out_result[0] + 1) / 2
                    save_image(out_result, '{}/Training Images/iteration_{}.{}'.format(self.base_path, iterations,
                                                                                       self.image_format))
                    self.gen.train()

                if save_model[0] and iterations % save_model[1] == 0:
                    self.save_checkpoint('checkpoint_iter_{}'.format(iterations), self.model_type)
                    average_loss.save('checkpoint_avg_loss', save_index=0)

                iterations += 1

            running_dis_loss /= (batches * 1.0)
            running_gen_loss /= (batches * 1.0)
            print('Epoch : {}, Generator Loss : {} and Discriminator Loss : {}'.format(i + 1, running_gen_loss,
                                                                                       running_dis_loss))
            save_tuple = ([running_gen_loss], [running_dis_loss])
            average_loss.add_loss(save_tuple)

        self.save_checkpoint('checkpoint_train_final', self.model_type)
        average_loss.save('checkpoint_avg_loss_final')

    def evaluate_model(self, loader, save_filename, no_of_images=1):
        # Considering that we have batch size of 1 for test set
        if self.gen is None or self.dis is None:
            raise Exception('Model has not been initialized and hence cannot be saved!');

        counter_images_generated = 0
        while counter_images_generated < no_of_images:
            gray, rgb = next(iter(loader))

            if self.device is not None:
                gray = gray.cuda()

            filename = '{}/Test Images/{}_{}.{}'.format(self.base_path, save_filename, self.count, self.image_format)
            real_filename = '{}/Test Images/{}_{}_real.{}'.format(self.base_path, save_filename, self.count,
                                                                  self.image_format)
            real_gray_filename = '{}/Test Images/{}_{}_real_gray.{}'.format(self.base_path, save_filename, self.count,
                                                                            self.image_format)
            self.count += 1

            self.gen.eval()
            out = self.gen(gray)
            out = out[0].detach().cpu()
            out = (out + 1) / 2
            save_image(out, filename)

            gray_img = gray[0].detach().cpu()
            save_image(gray_img, real_gray_filename)

            real_img = (rgb[0].detach().cpu() + 1) / 2
            save_image(real_img, real_filename)

            counter_images_generated += 1

    def evaluate_L1_loss_dataset(self, loader, train=False):

        if self.gen is None or self.dis is None:
            raise Exception('Model has not been initialized and hence cannot be evaluated!')

        loss_function = nn.L1Loss()
        self.gen.eval()
        total_loss = 0.0;
        iterations = 0;
        for gray, real in loader:
            iterations += 1
            if self.device is not None:
                gray = gray.cuda()
                real = real.cuda()

            gen_out = self.gen(gray)
            iteration_loss = loss_function(gen_out, real)
            total_loss += iteration_loss.item()
        total_loss = total_loss / (iterations * 1.0)
        train_test = 'test'
        if train:
            train_test = 'train'
        print('Total L1 loss over {} set is : {}'.format(train_test, total_loss))
        return total_loss;

    def change_params(self, epochs=None, learning_rate=None, leaky_relu=None, betas=None, lamda=None):
        if epochs is not None:
            self.epochs = epochs
            print('Changed the number of epochs to {}!'.format(self.epochs))
        if learning_rate is not None:
            self.lr = learning_rate
            print('Changed the learning rate to {}!'.format(self.lr))
        if leaky_relu is not None:
            self.leaky_relu_threshold = leaky_relu
            print('Changed the threshold for leaky relu to {}!'.format(self.leaky_relu_threshold))
        if betas is not None:
            self.betas = betas
            print('Changed the betas for Adams Optimizer!')
        if betas is not None or learning_rate is not None:
            self.gen_optim = optim.Adam(self.gen.parameters(), lr=self.lr, betas=self.betas)
            self.dis_optim = optim.Adam(self.dis.parameters(), lr=self.lr, betas=self.betas)

        if lamda is not None:
            self.lamda = lamda
            print('Lamda value has been changed to {}!'.format(self.lamda))

    def save_checkpoint(self, filename, model_type='unet'):
        if self.gen is None or self.dis is None:
            raise Exception('The model has not been initialized and hence cannot be saved !')

        filename = '{}/checkpoints/{}.pth'.format(self.base_path, filename)
        save_dict = {'model_type': model_type, 'dis_dict': self.dis.state_dict(), 'gen_dict': self.gen.state_dict(),
                     'lr': self.lr,
                     'epochs': self.epochs, 'betas': self.betas, 'image_size': self.image_size,
                     'leaky_relu_thresh': self.leaky_relu_threshold, 'lamda': self.lamda, 'base_path': self.base_path,
                     'count': self.count, 'image_format': self.image_format, 'device': self.device}

        torch.save(save_dict, filename)

        print('The model checkpoint has been saved !')

    def load_checkpoint(self, filename):
        filename = '{}/checkpoints/{}.pth'.format(self.base_path, filename)
        if not pathlib.Path(filename).exists():
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
        self.device = save_dict['device']
        device = self.get_device()
        if device is not self.device:
            error_msg = ''
            if self.device is None:
                error_msg = 'The model was trained on CPU and will therefore be continued on CPU only!'
            else:
                error_msg = 'The model was trained on GPU and cannot be loaded on a CPU machine!'
                raise Exception(error_msg)

        self.initialize_model(model_type=save_dict['model_type'])

        self.gen.load_state_dict(save_dict['gen_dict'])
        self.dis.load_state_dict(save_dict['dis_dict'])

        print('The model checkpoint has been restored!')
