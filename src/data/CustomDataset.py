"""
@author: harsh
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import pathlib
from PIL import Image
from skimage import color


def load_image(file_path):
    with open(file_path, 'rb') as file:
        with Image.open(file) as img:
            return img.convert('RGB')


class CustomDataset(Dataset):
    def __init__(self, path, image_size, image_format='png'):
        self.root = path
        self.image_size = image_size

        path_loc = pathlib.Path(path)
        if not path_loc.exists():
            raise Exception('The path provided is incorrect!')

        searchstring = os.path.join(path, '*.' + image_format)
        list_of_images = glob.glob(searchstring)
        list_of_images.sort()
        self.image_paths = list_of_images

    def __getitem__(self, index):
        file_path = self.image_paths[index]
        img = load_image(file_path)
        img = img.resize((self.image_size, self.image_size))
        img_np = np.array(img)

        # Scale the values to range -1 to 1
        img_np = (img_np - 127.5) / 127.5

        lab_img = color.rgb2lab(img)

        img_np = np.transpose(img_np, (2, 0, 1))
        lab_img = np.transpose(lab_img, (2, 0, 1))

        img_l = lab_img[0, :, :] / 100
        size = img_l.shape

        orig_img = torch.FloatTensor(img_np)
        gray_img = torch.FloatTensor(img_l).view(-1, size[0], size[1])
        return gray_img, orig_img

    def __len__(self):
        return len(self.image_paths)

    def __str__(self):
        return 'Dataset details are - \nRoot Location : {}\nImage Size : {}\nSize : {}'.format(
            self.root, self.image_size, self.__len__())
