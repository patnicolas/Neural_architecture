__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from models.gan.dcgan import DCGan
from models.gan.dcdiscriminator import DCDiscriminator
from models.gan.dcgenerator import DCGenerator
from models.gan.gan import Gan
from models.cnn.snneuralblock import SNNeuralBlock
from models.nnet.hyperparams import HyperParams

"""
    Spectral norm deep convolutional GAN. There are two constructors:
    - __init__ Direct instantiation from pre-defined generator and discriminator
    - transposed_initi  Mirrored instantiation for which the generator is a mirror de-convolutional network to the convolutional
        network used in the discriminator (auto_build)

    :param model_id: Identifier for the GAN
    :param gen: Deep de-convolutional generator
    :param disc: Deep convolutional discriminator
    :param hyper_params: Hyper-parameters used for training and evaluation
"""


class SNGan(DCGan):
    def __init__(self,
                 model_id: str,
                 gen: DCGenerator,
                 disc: DCDiscriminator,
                 hyper_params: HyperParams,
                 debug = None):
        super(SNGan, self).__init__(model_id, gen, disc, hyper_params, debug)


    @classmethod
    def transposed_init(cls,
                        model_id: str,
                        conv_dimension: int,
                        conv_neural_blocks: list,
                        output_gen_activation: torch.nn.Module,
                        in_channels: int,
                        out_channels: int,
                        hyper_params: HyperParams,
                        debug = None) -> Gan:
        """
            Generate a Spectral norm deep convolutional GAN using mirrors neural blocks
            :param model_id: Identifier for this deep convolutional GAN
            :param conv_dimension: Dimension of the convolution (1 or 2)
            :param conv_neural_blocks: Neural block used in the convolutional model of discriminator
            :param output_gen_activation: Activation function for the output to the generator
            :param in_channels: Number of input_tensor channels for the convolutional discriminator
            :param out_channels: Number of output channels for the convolutional discriminator
            :param hyper_params: Hyper-parameters used for training and evaluation
            :return: Deep convolutional GAN using a spectral norm
        """
        assert len(conv_neural_blocks) > 0, \
            f'SNGan Number of convolutional blocks {len(conv_neural_blocks)} should be > 0'

        for conv_neural_block in conv_neural_blocks:
            assert isinstance(conv_neural_block, SNNeuralBlock), f'SNGan Found non spectral norm neural block'

        return DCGan.transposed_init(model_id,
                                     conv_dimension,
                                     conv_neural_blocks,
                                     output_gen_activation,
                                     in_channels,
                                     out_channels,
                                     hyper_params,
                                     unsqueeze = True,
                                     debug = debug)
