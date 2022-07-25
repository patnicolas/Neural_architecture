__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from models.cnn.convneuralblock import ConvNeuralBlock
from models.cnn.deconvneuralblock import DeConvNeuralBlock
from models.cnn.convmodel import ConvModel
from models.cnn.deconvmodel import DeConvModel
from models.autoencoder.variationalneuralblock import VariationalNeuralBlock
from models.autoencoder.convvaemodel import ConvVAEModel

"""
    Configuration of convolutional neural block for the variational auto-encoder
    :param conv_dimension: Dimension of the convolution (1d or 2D)
    :param kernel_size: Size of the kernel for this convolutional block
    :param stride: Stride for this convolutional block
    :param padding: Padding for this convolutional block
    :param max_pooling_kernel: Size of the kernel used in max-pooling
    :param activation: Activation function for this convolutional block
    :param batch_normalization: Batch normalization for this convolutiona block if True
"""


class ConvVAEBlockConfig(object):
    def __init__(self,
                 conv_dimension: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 max_pooling_kernel: int,
                 activation: torch.nn.Module,
                 batch_normalization: bool):
        self.dim = conv_dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.max_pooling_kernel = max_pooling_kernel
        self.batch_normalization = batch_normalization


    def build(self, in_channels: int, out_channels: int, is_de_conv: bool):
        """
            Build a Convolutional neural block if is_de_conv is True or a de-convolutional block if False
            :param in_channels: Number of input_tensor channels (encoder) for this neural block
            :param out_channels:  Number of output channels for this block
            :param is_de_conv: Specify if this is a de-convolutional block
            :return: Convolutional or de-convolutional neural block
        """
        return DeConvNeuralBlock(self.dim,
                                 in_channels,
                                 out_channels,
                                 self.kernel_size,
                                 self.stride,
                                 self.padding,
                                 self.batch_normalization,
                                 self.activation,
                                 False) if is_de_conv else ConvNeuralBlock(self.dim,
                                                                           in_channels,
                                                                           out_channels,
                                                                           self.kernel_size,
                                                                           self.stride,
                                                                           self.padding,
                                                                           self.batch_normalization,
                                                                           self.max_pooling_kernel,
                                                                           self.activation,
                                                                           False,
                                                                           False)

"""
    Semi-automated configuration of Convolutional variational auto-encoder, using static method.
"""


class ConvVAEConfig(object):

    @staticmethod
    def build_network(conv_vae_block_configs: list, in_channels: int, hidden_dim: int, out_channels, is_de_conv: bool):
        """
            Build a decoder or encoder given a sequence of configuration for convolutional neural blocks
            :param conv_vae_block_configs: Sequence of configuration for convolutional neural blocks
            :param in_channels: Number of input_tensor channel for this encoder or decoder (input_tensor channels for
            :param hidden_dim: Size of hidden layers (with cascading sizing)
            :param out_channels: Number of output channels for the last convolutional (or de-convolutional layer)
            :param is_de_conv: Flag that specifies whether this is a de-convolutional network (True) or convolutional
                network (False)
            :return: Fully configured encoder or decoder
        """
        blocks = []
        input_channels = in_channels
        output_channels = hidden_dim
        for idx, block_config in enumerate(conv_vae_block_configs):
            blocks.append(block_config.transposed_init(input_channels, output_channels, is_de_conv))
            input_channels = output_channels
            if idx == len(conv_vae_block_configs) - 2:
                output_channels = out_channels
            elif is_de_conv:
                output_channels = output_channels//2
            else:
                output_channels = output_channels*2
        return blocks

    @staticmethod
    def build(
            model_id: str,
            conv_dim: int,
            conv_vae_block_configs: list,
            de_conv_vae_block_configs: list,
            in_channels: int,
            hidden_dim: int,
            out_channels,
            flatten_input : int,
            fc_hidden_dim: int,
            latent_size: int):
        """
            Build a convolutional variational auto-encoder using pre-defined configuration for symmetrical encoder and
            decoder. The in-channels and output_dim order are reversed for deconvolution (Decoder)
            :param model_id: Identifier for the model
            :param conv_dim: Dimension of the convolution {1D, 2D, 3D}
            :param conv_vae_block_configs: Configuration of the sequence of convolutional VAE blocks
            :param de_conv_vae_block_configs: Configuration of the sequence of de-convolutional VAE blocks
            :param in_channels: Number of input_tensor channels
            :param hidden_dim: Size of the hidden layer
            :param out_channels: Number of output channels (convolutional)
            :param flatten_input: Size of the flattened input_tensor to variational block (fully connected)
            :param fc_hidden_dim: Size of hidden layer of variational block
            :param latent_size: Dimension of the latent space
            :return: Convolutional variational auto-encoder model
        """
        conv_blocks = ConvVAEConfig.build_network(conv_vae_block_configs, in_channels, hidden_dim, out_channels, False)
        conv_model = ConvModel(f'{model_id}-encoder', conv_dim, conv_blocks, None)
        de_conv_blocks = ConvVAEConfig.build_network(de_conv_vae_block_configs, out_channels, hidden_dim * 2, in_channels, True)
        de_conv_model = DeConvModel(f'{model_id}-decoder', conv_dim, de_conv_blocks)
        variational_block = VariationalNeuralBlock(flatten_input, fc_hidden_dim, latent_size)

        return ConvVAEModel(f'{model_id} convolutional VAE', conv_model, de_conv_model, variational_block)







