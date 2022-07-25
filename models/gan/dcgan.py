__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from models.gan.gan import Gan
from models.cnn.convmodel import ConvModel
from models.cnn import DeConvModel
from models.gan.dcdiscriminator import DCDiscriminator
from models.gan.dcgenerator import DCGenerator
from models.nnet.hyperparams import HyperParams


"""
    Instantiation of deep convolutional GAN. There are two constructors:
    - Direct instantiation from pre-defined generator and discriminator (__init__)
    - Mirrored instantiation for which the generator is a mirror de-convolutional network to the convolutional
        network used in the discriminator (auto_build)
        
    :param model_id: Identifier for the GAN
    :param gen: GAN generator
    :param disc: Deep convolutional GAN discriminator
    :param hyper_params: Hyper-parameters used for training and evaluation
    :param debug: Debugging function
"""


class DCGan(Gan):
    def __init__(self, model_id: str, gen: DCGenerator, disc: DCDiscriminator, hyper_params: HyperParams, debug = None):
        super(DCGan, self).__init__(model_id, gen, disc, hyper_params, debug)

    @classmethod
    def transposed_init(cls,
                        model_id: str,
                        conv_dimension: int,
                        conv_neural_blocks: list,
                        output_gen_activation:  torch.nn.Module,
                        in_channels: int,
                        out_channels: int,
                        hyper_params: HyperParams,
                        unsqueeze: bool,
                        debug = None) -> Gan:
        """
            Generate a Deep convolutional GAN using mirrors neural blocks
            :param model_id: Identifier for this deep convolutional GAN
            :param conv_dimension: Dimension of the convolution (1 or 2)
            :param conv_neural_blocks: Neural block used in the convolutional model of discriminator
            :param output_gen_activation: Activation function for the output to the generator
            :param in_channels: Number of input_tensor channels for the convolutional discriminator
            :param out_channels: Number of output channels for the convolutional discriminator
            :param hyper_params: Hyper-parameters used for training and evaluation
            :param unsqueeze: Flag to specify if
            :param debug: Optional debugging functions
            :return: Deep convolutional GAN
        """
        conv_model = ConvModel.build(f'{model_id}-disc', conv_dimension, conv_neural_blocks, None)
        de_conv_model = DeConvModel.build_for_gan(f'{model_id}-gen',
                                                  conv_neural_blocks,
                                                  in_channels,
                                                  out_channels,
                                                  output_gen_activation)

        dc_discriminator = DCDiscriminator(conv_model)
        dc_generator = DCGenerator(de_conv_model, in_channels, unsqueeze)
        return cls(model_id, dc_generator, dc_discriminator, hyper_params)

    def save(self):
        """
            Save the parameters and convolutional models associated with the generator and discriminator
        """
        self.gen.save()
        self.disc.save()

    @staticmethod
    def load(gen_model_id: str, disc_model_id) -> (DCDiscriminator, DCGenerator):
        """
            Load the convolutional generator and discriminator for this GAN
            :param gen_model_id: Identifier for the convolutional generator
            :param disc_model_id: Identifier for the convolutional discriminator
            :return: Pair Generator, Discriminator
        """
        DCDiscriminator.load(disc_model_id), DCGenerator.load(gen_model_id)





