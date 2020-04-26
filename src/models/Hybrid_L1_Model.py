import torch.nn as nn
from torchvision.utils import save_image
import torch
import torchvision.models as models

from models.Model import Model


class Hybrid_L1_Model(Model):
    def __init__(self, base_path='', epochs=10, learning_rate=0.0002, image_size=256, leaky_relu=0.2,
                 betas=(0.5, 0.999), lamda=100, image_format='png'):
        super().__init__(base_path, epochs, learning_rate, image_size, leaky_relu, betas, lamda, image_format)

    def train_model(self, trainloader, average_loss, eval=(False, None, None), save_model=(False, 25),
                    display_test_image=(False, None, 25)):
        print('We will be using L1 loss with perpetual loss (L2)!')
        mean_loss = nn.BCELoss()
        l1_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()
        vgg16 = models.vgg16()
        vgg16_conv = nn.Sequential(*list(vgg16.children())[:-3])

        self.gen.train()
        self.dis.train()

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
                self.evaluate_L1_loss_dataset(trainloader, train=True)
                self.gen.train()

            running_gen_loss = 0
            running_dis_loss = 0

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
                gen_pre_train = mse_loss(vgg16_conv(fake_img), vgg16_conv(real_img))
                total_gen_loss = gen_adv_loss + self.lamda * gen_l1_loss + self.lamda * gen_pre_train
                total_gen_loss.backward()
                self.gen_optim.step()

                running_dis_loss += total_dis_loss.item()
                running_gen_loss += total_gen_loss.item()

            running_dis_loss /= (batches * 1.0)
            running_gen_loss /= (batches * 1.0)
            print('Epoch : {}, Generator Loss : {} and Discriminator Loss : {}'.format(i + 1, running_gen_loss,
                                                                                       running_dis_loss))
            if display_test_image[0] and i % display_test_image[2] == 0:
                self.gen.eval()
                out_result = self.gen(sample_img_test)
                out_result = out_result.detach().cpu()
                out_result = (out_result[0] + 1) / 2
                save_image(out_result, '{}/Training Images/epoch_{}.{}'.format(self.base_path, i,
                                                                               self.image_format))
                self.gen.train()

            save_tuple = ([running_gen_loss], [running_dis_loss])
            average_loss.add_loss(save_tuple)

            if save_model[0] and i % save_model[1] == 0:
                self.save_checkpoint('checkpoint_epoch_{}'.format(i), self.model_type)
                average_loss.save('checkpoint_avg_loss', save_index=0)

            self.lr_schedule_gen.step()
            self.lr_schedule_dis.step()
            for param_grp in self.dis_optim.param_groups:
                print('Learning rate after {} epochs is : {}'.format(i + 1, param_grp['lr']))

        self.save_checkpoint('checkpoint_train_final', self.model_type)
        average_loss.save('checkpoint_avg_loss_final', save_index=0)