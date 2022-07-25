__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from torch import nn

"""
    Define the layer for Deep Feed Forward network
    Components:
         Linear
         Activation
         Dropout (Optional)
         
    :param in_channels: Size of input_tensor encoder
    :type in_channels: int
    :param output_dim: Size of the output layer (or hidden layer)
    :type output_dim: int
    :param activation: Activation function associated with this layer
    :type activation: nn.Module
    :param drop_out: Ratio of neuron to be dropped out for regularization. drop_out = 0.0 means to regularization
    :type drop_out: float
"""


class DFFNeuralBlock(nn.Module):
    def __init__(self, input_size: int, output_sz: int, activation: nn.Module, drop_out: float = 0.0):
        super(DFFNeuralBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_sz
        self.modules = []
        self.modules.append(nn.Linear(input_size, output_sz, False))
        if activation is not None:
            self.modules.append(activation)
        # Only if regularization is needed
        if drop_out > 0.0:
            self.modules.append(nn.Dropout(drop_out))

    def __repr__(self) -> str:
        conf_repr = ' '.join([f'\n{str(module.detach())}' for module in self.modules])
        return f'Input layer size: {self.input_size}\nOutput layer size: {self.output_size}\n{conf_repr}'

    def get_modules(self) -> list:
        return self.modules

    def reset_parameters(self):
        self.linear.reset_parameters()


