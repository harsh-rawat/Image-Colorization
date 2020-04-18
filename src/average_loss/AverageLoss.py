from src.average_loss.Loss import *
import pickle
import matplotlib.pyplot as plt


class AverageLoss:
    def __init__(self, base_path):
        self.gen = Loss()
        self.dis = Loss()
        self.index = 0
        self.base_path = base_path

    # Assuming order - gen, dis
    def add_loss(self, losses):
        self.gen.add(losses[0])
        self.dis.add(losses[1])

    def plot(self, iteration_factor=10):
        plt.plot(self.gen.get_loss(), label='Generator')
        plt.plot(self.dis.get_loss(), label='Discriminator')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('{}/loss_curve_{}.png'.format(self.base_path, self.index))
        plt.show()

    def save(self, filename):
        save_dict = {'gen': self.gen, 'dis': self.dis, 'index': self.index, 'base_path': self.base_path}
        filepath = '{}/Loss_Checkpoints/{}_{}'.format(self.base_path, filename, self.index)
        with open(filepath, 'wb') as file:
            pickle.dump(save_dict, file)
        print('Losses have been saved!')
        self.index += 1

    def load(self, filename, index):
        filepath = '{}/Loss_Checkpoints/{}_{}'.format(self.base_path, filename, index)
        with open(filepath, 'rb') as file:
            save_dict = pickle.load(file)
            self.gen = save_dict['gen']
            self.dis = save_dict['dis']
            self.index = save_dict['index']
            self.base_path = save_dict['base_path']
        print('Checkpoint has been restored!')
