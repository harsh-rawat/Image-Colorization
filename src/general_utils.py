import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.utils import save_image

from data.Dataloader import Dataloader
from data.CustomDataset import CustomDataset


def generate_sample(loaders):
    train_imgs = next(iter(loaders))
    print('Shape of Grayscale Image Tensor is : {}'.format(train_imgs[0].shape))
    print('Shape of Colored Image Tensor is : {}'.format(train_imgs[1].shape))
    train_img_to_show_bw = train_imgs[0][0][0]
    train_img_to_show_rgb = train_imgs[1][0]
    train_img_to_show_rgb = np.transpose((train_img_to_show_rgb + 1) / 2, (1, 2, 0))

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(train_img_to_show_rgb)
    axes[1].imshow(train_img_to_show_bw, cmap='gray')
    plt.show()


def generate_loss_chart(labels, train, test):
    n = np.arange(len(labels))
    width = 0.3

    if len(labels) != len(train) != len(test):
        raise Exception('Length of the arrays do not match!')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(n, train, width, color='g', label='Loss over training set')
    ax.bar(n + width, test, width, color='y', label='Loss over test set')

    ax.set_ylabel('Loss')
    ax.set_xticks(n + width / 2)
    ax.set_xticklabels(labels)

    ax.legend()

    plt.savefig('compare_loss.png')

    plt.show()


def convert_to_grayscale(path_to_folder, global_path, folder_name, save_colored=False, image_size=256,
                         image_format='jpg'):
    loader = Dataloader(path_to_folder, image_size, batch_size=1, image_format=image_format,
                        validation_required=(False, 0.2, 'train_valid_split'))
    train_loader, valid_loader = loader.get_data_loader()
    index = 1;
    base_path = ''
    base_path_color = ''
    try:
        converted_path = '{}/{}'.format(global_path, folder_name)
        converted_path_color = '{}/{}_color'.format(global_path, folder_name)
        if not os.path.exists(converted_path):
            os.makedirs(converted_path)
        if not os.path.exists(converted_path_color) and save_colored:
            os.makedirs(converted_path_color)
        base_path = converted_path
        base_path_color = converted_path_color
    except OSError:
        print('Error: Creating directory of data')

    for gray, color in train_loader:
        save_name = '{}/{}.{}'.format(base_path, index, image_format)
        save_name_color = '{}/{}.{}'.format(base_path_color, index, image_format)
        index = index + 1
        save_image(gray, save_name)
        if save_colored:
            save_image((color + 1) / 2, save_name_color)
    print('Task Completed!')


def calculate_closeness_score(path_data, folder_real, folder_test, image_size=256, image_format='jpg'):
    real_dataset = CustomDataset('{}/{}'.format(path_data, folder_real), image_size, image_format, image_type='rgb')
    test_dataset = CustomDataset('{}/{}'.format(path_data, folder_test), image_size, image_format, image_type='rgb')
    length = len(real_dataset)

    total_loss = 0.0;
    for index in range(length):
        real_img = real_dataset[index][1]
        test_img = test_dataset[index][1]
        iteration_loss = torch.mean(torch.abs(real_img - test_img))
        total_loss += iteration_loss.item()
    total_loss = total_loss / (length * 1.0)
    print('Closeness score is : {}'.format(total_loss))
    return total_loss;


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='General Utils for this project!')
    parser.add_argument('-dpath', metavar='dataset-path', action='store', required=True,
                        help='The base path of the dataset folder')
    parser.add_argument('-bpath', metavar='base-path', action='store', default=None,
                        help='The path of the working directory')
    parser.add_argument('-folder', metavar='folder', action='append', required=True,
                        help='Folder name')
    parser.add_argument('-format', metavar='Image Format', action='store', default='jpg',
                        help='Image format to be considered')
    parser.add_argument('-size', metavar='Image Size', action='store', default=256, help='Image size to be considered')
    parser.add_argument('-convert_to_grayscale', action='store_true', default=False, help='Use this option to convert '
                                                                                          'all colored images in a '
                                                                                          'fdlder to grayscale')
    parser.add_argument('-find_closeness_score', action='store_true', default=False, help='Find closeness score')
    args = parser.parse_args()

    if args.convert_to_grayscale:
        if args.bpath is None:
            raise Exception('Base Path not provided!')
        convert_to_grayscale(args.dpath, args.bpath, args.folder[0], True, args.size, args.format)
    if args.find_closeness_score:
        if len(args.folder) != 2:
            raise Exception('Please provide folder for real images and generated images')
        calculate_closeness_score(args.dpath, args.folder[0], args.folder[1], args.size, args.format)
