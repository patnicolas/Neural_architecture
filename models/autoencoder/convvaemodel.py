__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import constants
from models.cnn.convmodel import ConvModel
from models.cnn.deconvmodel import DeConvModel
from models.autoencoder.variationalneuralblock import VariationalNeuralBlock

"""
    Convolutional model for the variational auto-encoder
    There are two constructors:
    __init__   with predefined convolutional, de-convolutional and variational components
    transpose_init For which the de-convolutional and variational components are automatically
    generated from a list of convolutional neural blocks
    
    :param model_id: Identifier for the convolutional variational auto-encoder_model
    :param encoder_model: Convolutional model as encoder_model for this auto-encoder_model
    :param decoder_model: De-convolutional model as Decoder for this auto-encoder_model
    :param variational_block: Variational components
"""


class ConvVAEModel(torch.nn.Module):
    def __init__(self,
                 model_id: str,
                 encoder_model: ConvModel,
                 decoder_model: DeConvModel,
                 variational_block: VariationalNeuralBlock):
        super(ConvVAEModel, self).__init__()
        self.model_id = model_id
        self.encoder_model = encoder_model.conv_model
        self.decoder_model = decoder_model.de_conv_model
        self.variational_block = variational_block
        self.input_channels = encoder_model.input_size

    @classmethod
    def transposed_init(cls,
                        model_id: str,
                        conv_dimension: int,
                        conv_neural_blocks: list) -> torch.nn.Module:
        """
            Generate a Deep convolutional GAN using mirrors neural blocks
            :param model_id: Identifier for this deep convolutional GAN
            :param conv_dimension: Dimension of the convolution (1 or 2)
            :param conv_neural_blocks: Neural block used in the convolutional model of discriminator
            :return: Convolutional variational auto-encoder
        """
        conv_model = ConvModel(f'{model_id}-disc', conv_dimension, conv_neural_blocks, None, True)
        de_conv_in_channels = conv_model.output_size // 2
        de_conv_model = DeConvModel.build_for_vae(f'{model_id}-gen', conv_neural_blocks, torch.nn.Sigmoid())
        hidden_dim = 44
        _variational_block = VariationalNeuralBlock(conv_model.output_size, hidden_dim, de_conv_in_channels)
        return cls(model_id, conv_model, de_conv_model, _variational_block)


    def __repr__(self):
        return f'Encoder:{repr(self.encoder_model)}\n\nVariational:\n{repr(self.variational_block)}\n\nDecoder:{repr(self.decoder_model)}'

    def flatten_conv_tensor_shape(self, x: torch.Tensor) -> list:
        """
            Extract the shape of the output input_tensor of the convolutional network
            :param x: Tensor input_tensor to the convolutional network
            :return: Shape of the unsqueeze output of the convolutional network
        """
        x = self.encoder_model(x)
        batch, _, _, _ = x.shape
        y = x.view(batch, -1)
        return list(y.shape)


    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
            Process the model as sequence of modules, implicitly called by __call__
            :param x: Input input_tensor
            :return: z, mean and log variance input_tensor
        """
        constants.log_size(x, 'ConvVAEModel enter encoder')
        x = self.encoder_model(x)
        constants.log_size(x, 'ConvVAEModel exit encoder')
        # shapes = list(x.shape)
        batch, a, b, c = x.shape
        x = x.view(batch, -1)
        constants.log_size(x, 'ConvVAEModel enter variational')
        z, mu, log_var = self.variational_block(x)
        constants.log_size(z, 'ConvVAEModel exit variational')
        z = z.view(batch, mu.shape[1], b, c)
        constants.log_size(z, 'ConvVAEModel Enter decoder')
        z = self.decoder_model(z)
        return z, mu, log_var
