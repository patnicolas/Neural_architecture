__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from models.gan.generator import Generator
from models.cnn.deconvnet import DeConvNet
from models.cnn.deconvmodel import DeConvModel
from models.nnet.neuralmodel import NeuralModel

"""
    Deep Convolutional Generator.
    There are two constructor:
    __init__   Generic constructor using a pre-build de-convolutional model
    build      Parameterized constructor using cascading size
    
    :param de_conv_model: De_convolutional neural network model
    :param in_channels: Size of latent space (number of latent encoder)
    :param unsqueeze: Boolean flag to specify whether the input_tensor noise input_tensor should be unsqueeze
"""


class DCGenerator(Generator):
    def __init__(self, de_conv_model: DeConvModel, z_dim: int, unsqueeze: bool):
        super(DCGenerator, self).__init__(de_conv_model, z_dim)
        self.unsqueeze = unsqueeze

    @classmethod
    def build(  cls,
                model_id: str,
                conv_dimension: int,
                z_dim: int,
                hidden_dim: int,
                out_dim: int,
                conv_params: list):
        """"
            Alternative constructor for generator using a de convolutional neural model
            The convolutional parameters are: kernel_size, stride, padding, batch_norm, activation.
            :param model_id: Identifier for the model used a generator
            :param conv_dimension: Dimension of the convolution (1 time series, 2 images, 3 video..)
            :param z_dim: Size of the latent space
            :param hidden_dim: Size of the intermediate blocks
            :param out_dim: Number of output channels
            :param conv_params: List of convolutional parameters {kernel_size, stride, padding, batch_norm,  activation}
            :returns: Instance of Gan generator
        """
        de_conv_model = DeConvNet.feature_extractor(model_id, conv_dimension, z_dim, hidden_dim, out_dim, conv_params)
        return cls(de_conv_model, z_dim)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
            Delegate the execution of the model to the child model
            :param noise: Input input_tensor
            :return: Tensor predicted by the child model
        """
        if self.unsqueeze:
            noise = noise.view(len(noise), self.z_dim, 1, 1)
        return self.model(noise)

    def _state_params(self) -> dict:
        return {"model_id":self.model_id,"conv_dimension":self.conv_dimension, "input_size":self.input_size,
                "output_size":self.output_size,"dff_model_input_size":self.dff_model_input_size}

    def save(self):
        """
            Save this (de)-convolutional generator into local file
        """
        extra_params = {"z_dim":self.z_dim}
        self.model.save(extra_params)

    @staticmethod
    def load(model_id: str) -> NeuralModel:
        return DeConvModel.load(model_id)
