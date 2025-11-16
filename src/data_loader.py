import os
import numpy as np
from torch import Tensor
from torch.utils import data
# import torchvision.transforms as transforms


class DataLoader:
    '''
    Loads "MNIST handwritten digits" dataset
    locally saved, downloaded from:
    https://www.kaggle.com/datasets/hichamachahboun/mnist-handwritten-digits
    '''
    _DATASET_DIR = 'mnist_handwritten_digits'

    @staticmethod
    def get_labels(is_train=True):
        return np.load(DataLoader._get_data_path(is_labels=True, is_train=is_train))
    
    @staticmethod
    def get_images(is_train=True):
        return np.load(DataLoader._get_data_path(is_labels=False, is_train=is_train))

    @staticmethod
    def _get_data_path(is_labels, is_train=True):
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(repo_dir, DataLoader._DATASET_DIR)
        file_path = f'{"train" if is_train else "test"}_{"labels" if is_labels else "images"}'
        return os.path.join(data_dir, file_path + '.npy')

    @staticmethod
    def build(is_train=True, batch_size=5, shuffle=True):
        images = DataLoader.get_images(is_train)
        # trans = transforms.Compose([transforms.ToTensor()])
        # images = trans(images)
        labels = DataLoader.get_labels(is_train)
        dataset = data.TensorDataset(Tensor(images), Tensor(labels).long())
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
