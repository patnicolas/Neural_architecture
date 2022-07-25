__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from torch import nn
from models.nnet.neuralmodel import NeuralModel
from models.dffn import DFFNet

"""
    Generic discriminator for Generative Adversarial Networks. 
    :param neural_model: Neural network model (i.e. Convolutional or Feed-forward model)
    :type neural_model: Class derived from nnet.neuralmodel.NeuralModel
"""


class Discriminator(nn.Module):
    def __init__(self, neural_model: NeuralModel):
        super(Discriminator, self).__init__()
        self.model = neural_model

    @classmethod
    def build(  cls,
                model_id: str,
                input_size: int,
                hidden_dim: int,
                output_size: int,
                dff_params:  list) -> NeuralModel:
        """
            Build a feed-forward network model as Gan discriminator
            :param model_id: Identifier for the decoder
            :param input_size: Size of the connected input_tensor layer
            :param hidden_dim: Size of the last hidden layer. The size of previous layers are halved from the
                           previous layer
            :param output_size: Size of the output layer
            :param dff_params: List of parameters tuple{ (activation_func, drop-out rate)
            :returns: Instance of Gan generator
        """
        dff_decoder_model = DFFNet.build_decoder(model_id, input_size, hidden_dim, output_size, dff_params)
        return cls(dff_decoder_model)

    def forward(self, x):
        output = self.model(x)
        return output

    def __repr__(self):
        return f'Discriminator: -------------{repr(self.model)}'