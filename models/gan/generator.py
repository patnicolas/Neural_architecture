__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import constants
from torch import nn
from models.nnet.neuralmodel import NeuralModel
from models.dffn import DFFNet

"""
    Generator using Feed-forward network model. There are three constructors
    - Default constructor using any give neural model 
    - Deep feed forward network build()
    The prediction is actually delegated to the child model provided in the constructor
    
    :param neural_model: Neural network model (i.e. Convolutional net, Feed-forward net)
    :param in_channels: Size of input_tensor layer
"""


class Generator(nn.Module):
    def __init__(self, neural_model: NeuralModel, z_dim: int):
        super(Generator, self).__init__()
        self.model = neural_model
        self.z_dim = z_dim

    @classmethod
    def build(cls,
              model_id: str,
              input_dim: int,
              hidden_dim: int,
              output_size: int,
              dff_params: list):
        """
            Build a feed-forward network model as Gan generator
            :param model_id: Identifier for the decoder
            :param input_dim: Size of the connected input_tensor layer
            :param hidden_dim: Size of the last hidden layer. The size of previous layers are halved from the
                       previous layer
            :param output_size: Size of the output layer
            :param dff_params: List of parameters tuple{ (activation_func, drop-out rate)
            :returns: Instance of Gan generator
        """
        dff_encoder_model = DFFNet.build_encoder(model_id, input_dim, hidden_dim, output_size, dff_params)
        return cls(dff_encoder_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Delegate the execution of the model to the child model
            :param x: Input input_tensor
            :return: Tensor predicted by the child model
        """
        return self.model(x)

    def __repr__(self):
        return f'Generator: ----------{repr(self.model)}'

    def noise(self, num_samples: int, unsqueeze_noise = None) -> torch.Tensor:
        """
            Create noise input_tensor for the latent space
            :param num_samples: Number of random samples from the latent space
            :param unsqueeze_noise: Function to unsqueeze the noise
            :returns: Torch input_tensor

        r =  torch.randn(num_samples, self.in_channels, device=constants.torch_device)
        if unsqueeze_noise is not None:
            return unsqueeze_noise(r, self.in_channels)
        else:
            return r
        """
        return Generator.adjusted_noise(num_samples, self.z_dim, unsqueeze_noise)


    @staticmethod
    def adjusted_noise(num_samples: int, z_dim: int, unsqueeze_noise = None) -> torch.Tensor:
        """
            Create noise input_tensor for the latent space
            :param num_samples: Number of random samples from the latent space
            :param z_dim: Size of the z_space
            :param unsqueeze_noise: Function to unsqueeze the noise
            :returns: Torch input_tensor
        """
        r =  torch.randn(num_samples, z_dim, device=constants.torch_device)
        return unsqueeze_noise(r, z_dim) if unsqueeze_noise is not None else r

