import torch
from src.data.CustomDataset import CustomDataset
from random import shuffle
import pickle
import pathlib


class Dataloader:
    def __init__(self, path, image_size, batch_size=16, image_format='png',
                 validation_required=(False, 0.2, 'train_valid_split')):
        self.path = path
        self.image_size = image_size
        self.image_format = image_format
        self.validation_req = validation_required
        self.batch_size = batch_size

    def get_data_loader(self, load_indexes=True, save_indexes=False):
        dataset = CustomDataset(self.path, self.image_size, self.image_format)
        print(dataset)
        size = len(dataset)

        train_index = list(range(size))
        valid_index = []

        if self.validation_req[0]:
            file_path = '{}/{}'.format(self.path, self.validation_req[2])
            path = pathlib.Path(file_path)
            if path.exists() and load_indexes:
                with open(file_path, 'rb') as file:
                    save_dict = pickle.load(file)
                    train_index = save_dict['train']
                    valid_index = save_dict['valid']
                    print('Index files have been loaded!')
            else:
                train_size = int(size * (1 - self.validation_req[1]))
                indexes = list(range(size))
                shuffle(indexes)
                train_index = indexes[0:train_size]
                valid_index = indexes[train_size:]

                if save_indexes:
                    with open(file_path, 'wb') as file:
                        save_dict = {'train': train_index, 'valid': valid_index}
                        pickle.dump(save_dict, file)
                        print('Index files have been saved!')

        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_index)

        trainloader = None
        validloader = None
        if len(valid_index) == 0:
            trainloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=self.batch_size)
        else:
            trainloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=self.batch_size)
            validloader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=1)

        return trainloader, validloader
