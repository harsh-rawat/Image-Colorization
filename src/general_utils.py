import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from data.Dataloader import Dataloader


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


def convert_to_grayscale(path_to_folder, global_path, folder_name,  image_size=256, image_format='jpg'):
    loader = Dataloader(path_to_folder, image_size, batch_size=1, image_format=image_format)
    train_loader, valid_loader = loader.get_data_loader()
    index = 1;
    base_path = ''
    try:
        converted_path = '{}/{}'.format(global_path, folder_name)
        if not os.path.exists(converted_path):
            os.makedirs(converted_path)
        base_path = converted_path
    except OSError:
        print('Error: Creating directory of data')

    for gray, color in train_loader:
        save_name = '{}/{}.{}'.format(base_path, index,image_format)
        index = index + 1
        save_image(gray, save_name)
    print('Task Completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='General Utils for this project!')
    parser.add_argument('-dpath', metavar='dataset-path', action='store', required=True,
                        help='The base path of the dataset folder')
    parser.add_argument('-bpath', metavar='base-path', action='store', required=True,
                        help='The path of the working directory')
    parser.add_argument('-folder', metavar='folder', action='store', required=True,
                        help='Folder name')
    parser.add_argument('-format', metavar='Image Format', action='store', default='jpg',
                        help='Image format to be considered')
    parser.add_argument('-size', metavar='Image Size', action='store', default=256, help='Image size to be considered')
    parser.add_argument('-convert_to_grayscale', action='store_true', default=False, help='Use this option to convert '
                                                                                          'all colored images in a '
                                                                                          'fdlder to grayscale')

    args = parser.parse_args()

    if args.convert_to_grayscale:
        convert_to_grayscale(args.dpath, args.bpath, args.folder, args.size, args.format)