__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import constants
from torch.autograd import Variable

"""
    Variational neural block used to implement the variational connections with 4 components
    Full connected flattening layers:
    -   Mean layer 
    -   log-variance layer
    -   sampling layer
    :param in_channels; Size of the flattening layer
    :param hidden_dim: Size of the input_tensor layer for the mean and variance layers
    :param latent_size: Size/num_tfidf_features of the latent space
"""


class VariationalNeuralBlock(torch.nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, latent_size: int):
        super(VariationalNeuralBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, latent_size)
        self.log_var = torch.nn.Linear(hidden_dim, latent_size)
        self.sampler_fc = torch.nn.Linear(latent_size, latent_size)

    def __repr__(self):
        return f'   {repr(self.fc)}\n   {repr(self.mu)}\n   {repr(self.log_var)}\n   {repr(self.sampler_fc)}'

    def in_channels(self):
        return self.fc.in_features

    @classmethod
    def re_parametrize(cls, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        random sample the z-space using the mean and log variance
        :param mu: Mean of the distribution in the z-space (latent)
        :param log_var: Logarithm of the variance of the distribution in the z-space
        :return: Sampled data point from z-space
        """
        std = log_var.mul(0.5).exp_()
        std_dev = std.data.new(std.size()).normal_()
        eps = Variable(std_dev)
        return eps.mul_(std).add_(mu)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
            Process the model as sequence of modules, implicitly called by __call__
            :param x: Input input_tensor as flattened output input_tensor from the convolutional layers
            :return: z, mean and log variance input_tensor
        """
        constants.log_size(x, 'input_tensor variational')
        x = self.fc(x)
        constants.log_size(x, 'fc variational')
        mu = self.mu(x)
        constants.log_size(mu, 'mu variational')
        log_var = self.log_var(x)
        z = VariationalNeuralBlock.re_parametrize(mu, log_var)
        constants.log_size(z, 'z variational')
        return self.sampler_fc(z), mu, log_var






