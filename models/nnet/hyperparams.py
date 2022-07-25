__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from torch import optim
from torch import nn
import constants

"""
    Generic class to encapsulate feature_name configuration parameters of training of any Neural Networks. 
    Not all these parameters are to be tuned during training..  The label reshaping function is
    a not tunable parameters used to width label input_tensor if needed
    
    :param learning_rate: Learning rate for the selected optimizer
    :type learning_rate: float
    :param momentum: Momentum rate for the selected optimizer
    :type momentum: float
    :param epochs: Number of epochs or iterations
    :type epochs: int
    :param optim_label: Label for the optimizer (i.e. 'sgd', 'adam', ...)
    :type optim_label: str
    :param bath_size: Size of the batch used for the batch normalization
    :type batch_size: int
    :param early_stop_patience: Ratio used for stopping training
    :type early_stop_patience: float
    :param loss_function: PyTorch nn.Module loss function (i.e BCELoss, MSELoss....)
    :type loss_function: torch.nn.Module
"""


class HyperParams(object):
    def __init__(self,
                 lr: float,
                 momentum: float,
                 epochs: int,
                 optim_label: str,
                 batch_size: int,
                 early_stop_patience: int,
                 loss_function: nn.Module,
                 drop_out: float,
                 normal_weight_initialization: bool = False):

        assert 1e-6 <= lr <= 0.1, f'Learning rate {lr} should be [1e-6, 0.1]'
        assert 3 <= epochs <= 500, f'Number of epochs {epochs} should be [3, 50]'
        assert 0.4 <= momentum <= 0.999, f'Context stride {momentum} should be [0.5, 0.999]'
        assert 2 <= early_stop_patience <= 16, f'Size of embeddings {early_stop_patience} should be [2, 16]'
        assert 1 <= batch_size <= 256, f'Size of batch {batch_size} should be [2, 256]'
        assert 0.0 <= drop_out <= 1.0, f'Drop out {drop_out} should be [0, 1]'

        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.momentum = momentum
        self.early_stop_patience = early_stop_patience
        self.normal_weight_initialization = normal_weight_initialization
        self.optim_label = optim_label
        self.drop_out = drop_out
        torch.manual_seed(42)

    @staticmethod
    def test_conversion(label_conversion_func) -> torch.Tensor:
        x = torch.rand(20)
        return label_conversion_func(x)

    def initialize_weight(self, modules: list):
        """
             In-place initialization weight of a list of linear module give a encoder_model
            :param modules: torch module to be initialize
        """
        if self.normal_weight_initialization is True:
            for module in modules:
                if type(module) == nn.Linear:
                    nn.init.normal_(module.weight)


    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
            Select the optimizer for generated from encoder_model parameters given the optimization label
            - SGD with nesterov momentum
            - Adam
            - Plain vanilla SGD
            :param model: ML model for which to extract parameters
            :return: Optimizer
        """
        if self.optim_label == constants.optim_adam_label:
            return optim.Adam(model.parameters(), self.learning_rate)
        elif self.optim_label == constants.optim_nesterov_label:
            return optim.SGD(
                model.parameters(),
                lr = self.learning_rate,
                momentum = self.momentum,
                nesterov = True)
        else:
            return optim.SGD(
                model.parameters(),
                lr = self.learning_rate,
                momentum=self.momentum,
                nesterov=False)


    def optimizer_tune(self, model: nn.Module, config):
        if self.optim_label == constants.optim_adam_label:
            return optim.Adam(model.parameters(), lr = config.get("learning_rate", 0.001),)
        elif self.optim_label == constants.optim_nesterov_label:
            return optim.SGD(
                model.parameters(),
                lr = config.get("learning_rate", 0.001),
                momentum = config.get("momentum", 0.95),
                nesterov = True)
        else:
            return optim.SGD(
                model.parameters(),
                lr = config.get("learning_rate", 0.001),
                momentum=config.get("momentum", 0.95),
                nesterov=False)

    def __repr__(self) -> str:
        return f'   Learning rate: {self.learning_rate}\n   Momentum: {self.momentum}\n   Number of epochs: {self.epochs}'\
                f'\n   Batch size: {self.batch_size}\n   Early stop patience: {self.early_stop_patience}' \
                f'\n   Optimizer: {self.optim_label}\n   Loss function: {repr(self.loss_function)}'

    def get_label(self) -> str:
        return f'lr.{self.learning_rate}-bs.{self.batch_size}'


    def grid_search(self, lrs: list, batch_sizes: list) -> iter:
        """
            Generate multiple Hyper-parameter using a list of learning rate and batch-size
            :param lrs:  List of learning rate
            :param batch_sizes:  Size of batch
            :return: Iterator for Hyper-params
        """
        return (HyperParams(
            lr,
            self.momentum,
            self.epochs,
            self.optim_label,
            batch_size,
            self.early_stop_patience,
            self.loss_function,
            self.drop_out,
            self.normal_weight_initialization) for lr in lrs for batch_size in batch_sizes)

