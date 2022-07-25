__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from models.gan.discriminator import Discriminator
from models.cnn.convnet import ConvModel
from models.cnn.convnet import ConvNet
from models.nnet.neuralmodel import NeuralModel

"""
    Deep Convolutional Discriminator.
    There are two constructor:
    __init__   Generic constructor using a pre-build convolutional model
    build      Parameterized constructor using cascading size

    :param conv_model: Convolutional neural network model
    :param in_channels: Size of latent space (number of latent encoder)
"""


class DCDiscriminator(Discriminator):
    def __init__(self, conv_model: ConvModel):
        super(DCDiscriminator, self).__init__(conv_model)

    @classmethod
    def build(  cls,
                model_id: str,
                conv_dimension: int,
                z_dim: int,
                hidden_dim: int,
                out_dim: int,
                conv_params: list):
        """
             Alternative constructor for discriminator using a convolutional neural model
             The convolutional parameters are: kernel_size, stride, padding, batch_norm, max_pooling_kernel, activation.
             :param model_id: Identifier for the model used a generator
             :param conv_dimension: Dimension of the convolution (1 time series, 2 images, 3 video..)
             :param z_dim: Size of the latent space
             :param hidden_dim: Size of the intermediate blocks
             :param out_dim: Number of output channels
             :param conv_params: List of convolutional parameters
                                {kernel_size, stride, padding, batch_norm, max_pooling_kernel, activation}
             :returns: Instance of Gan generator
         """
        conv_model = ConvNet.feature_extractor(model_id, conv_dimension, z_dim, hidden_dim, out_dim, conv_params)
        return cls(conv_model)

    def save(self):
        """
            Save this convolutional discriminator into local file
        """
        self.model.save(None)

    @staticmethod
    def load(model_id: str) -> NeuralModel:
        return ConvModel.load(model_id)