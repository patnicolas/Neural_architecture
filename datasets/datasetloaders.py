__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np

"""
    Wraps static methods to load public data sets. The methods generate two data loader
    - Training 
    - Evaluation
    
    :param batch_size: Size of the batch used in the loader
    :param num_samples: Number of samples loaded (or all data if num_samples <= 0)
    :param split_ratio: Training-validation random split ratio
"""


class DatasetLoaders(object):
    def __init__(self, batch_size: int, num_samples: int, split_ratio: float):
        assert batch_size >= 4, f'Batch size {batch_size} should be >= 4'
        assert 0.5 <= split_ratio <= 0.95, f'Training-validation split ratio {split_ratio} should be [0.5, 0.95]'

        self.batch_size = batch_size
        self.num_samples = num_samples
        self.split_ratio = split_ratio


    def load_mnist(self, norm_factors: list) -> (DataLoader, DataLoader):
        """
            Load MNIST library of digits
            :param norm_factors: List of two normalization factors
            :return: Pair of Data loader for training data and validation data
        """
        assert len(norm_factors) == 2, f'Number of normalization factors {len(norm_factors)} should be 2'

        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((norm_factors[0],), (norm_factors[1],)),
        ])
        mnist_dataset = MNIST('../data/', download=True, transform=transform)
        return self.__generate(mnist_dataset)


    def load_dataset(self, dataset: Dataset) -> (DataLoader, DataLoader):
        """
            Generate Training and evaluation data loader from a given dataset
            :param dataset: Dataset
            :return: Pair of Data loader for training data and validation data
        """
        return self.__generate(dataset)

    def load_tensor(self, data: torch.Tensor, norm_factors: list) -> (DataLoader, DataLoader):
        """
            Generate Training and evaluation data loader from a given input_tensor
            :param data: Input input_tensor
            :param norm_factors: Normalization factors
            :return: Pair of Data loader for training data and validation data
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((norm_factors[0],), (norm_factors[1],)),
        ])
        dataset = Dataset(data, transform)
        return self.__generate(dataset)

    # ------------------------ Supporting private method ------------------

    def __generate(self, dataset: Dataset):
        if self.num_samples > 0:
            indices = np.arange(self.num_samples)
            dataset = torch.utils.data.Subset(dataset, indices)

        training_size = int(len(dataset) * self.split_ratio)
        validation_size = len(dataset) - training_size
        train_dataset, valid_dataset = random_split(dataset, (training_size, validation_size))
        train_data_loader = DataLoader(train_dataset, batch_size= self.batch_size, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size= self.batch_size, shuffle=True)

        return train_data_loader, valid_data_loader
