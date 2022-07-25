__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from models.dffn.dffmodel import DFFModel
from models.autoencoder.variationalneuralblock import VariationalNeuralBlock
import constants

"""
    Feed-forward model for Variational auto-encoder
    :param model_id: Identifier for the feed-forward network model for the variational auto encoder
    :param encoder_model: Feed-forward model as encoder for the variational auto encoder
    :param decoder_model: Feed-forward model as decoder for the variational auto encoder
    :param variational_block: Variational neural block
"""


class DFFVAEModel(torch.nn.Module):
    def __init__(self,
                 model_id: str,
                 encoder_model: DFFModel,
                 decoder_model: DFFModel,
                 variational_block: VariationalNeuralBlock):
        super(DFFVAEModel, self).__init__()
        self.model_id = model_id
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.variational_block = variational_block
        self.input_channels = None

    def __repr__(self):
        return f'Encoder:{repr(self.encoder_model)}\nVariational model:\n{repr(self.variational_block)}\nDecoder: {repr(self.decoder_model)}'

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
           Process the model as sequence of modules, implicitly called by __call__
           :param x: Input input_tensor
           :return: z, mean and log variance input_tensor
           """
        constants.log_size(x, 'Input dff_vae')
        x = self.encoder_model(x)
        constants.log_size(x, 'after encoder_model')
        batch, a = x.shape
        x = x.view(batch, -1)
        constants.log_size(x, 'flattened')
        z, mu, log_var = self.variational_block(x)
        constants.log_size(z, 'after variational')
        constants.log_size(mu, 'mu')
        z = z.view(batch, a)
        constants.log_size(z, 'z.view')
        z = self.decoder_model(z)
        return z, mu, log_var

