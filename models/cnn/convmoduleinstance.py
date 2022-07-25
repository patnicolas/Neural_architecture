__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch

"""
    Object (static) to evaluate the property of a convolutional PyTorch module. All methods are static
"""


class ConvModuleInstance(object):
    @staticmethod
    def is_conv(module: torch.nn.Module) -> bool:
        """
            Test if this module is a convolutional layer
            :param module: Torch module
            :return: True if this is a convolutional layer, False otherwise
        """
        return isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv1d)

    @staticmethod
    def is_de_conv(module: torch.nn.Module) -> bool:
        """
             Test if this module is a de-convolutional layer
             :param module: Torch module
             :return: True if this is a de-convolutional layer, False otherwise
         """
        return isinstance(module, torch.nn.ConvTranspose2d) or isinstance(module, torch.nn.ConvTranspose1d)

    @staticmethod
    def is_batch_norm(module: torch.nn.Module) -> bool:
        """
             Test if this module is a batch normalization layer
             :param module: Torch module
             :return: True if this is a batch normalization layer, False otherwise
         """
        return isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d)

    @staticmethod
    def is_activation(module: torch.nn.Module):
        """
             Test if this module is an activation layer
             :param module: Torch module
             :return: True if this is a activation layer, False otherwise
         """
        return isinstance(module, torch.nn.ReLU) or \
               isinstance(module, torch.nn.LeakyReLU) or \
               isinstance(module, torch.nn.Tanh)

    @staticmethod
    def extract_conv_modules(conv_modules: list) -> \
            (torch.nn.Module, torch.nn.Module, torch.nn.Module):
        """
            Extract convolutional layer, batch normalization and activation function from a neural block
            :param conv_modules: Modules defined in this neural block
            :return: Tuple convolutional layer, batch normalization and activation function
        """
        activation_function = None
        conv_layer = None
        batch_norm_module = None
        for conv_module in conv_modules:
            if ConvModuleInstance.is_conv(conv_module):
                conv_layer = conv_module
            elif ConvModuleInstance.is_batch_norm(conv_module):
                batch_norm_module = conv_module
            elif ConvModuleInstance.is_activation(conv_module):
                activation_function = conv_module
        return conv_layer, batch_norm_module, activation_function
