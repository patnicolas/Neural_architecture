__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from torch import nn

from models.cnn.convmoduleinstance import ConvModuleInstance
from models.cnn.convneuralblock import ConvNeuralBlock

image_channels = 3

"""
    Generic De-convolutional Neural block or Convolution transpose block. Contrary to convolutional networks, 
    deconvolution does not support max polling as down-sampling does make any sense in this context.
    Components:
        Convolution transpose
        Batch normalization [Optional]
        Activation 
        
    Formula for processing neural block with in_channels
        output_dim = stride*(in_channels -1) - 2*padding + kernel_size
        
    :param conv_dim Dimension of the convolution (1 or 2)
    :param in_channels Number of input_tensor channels
    :param output_dim Number of output channels
    :param kernel_size Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
    :param stride Stride for convolution (st) for 1D, (st, st) for 2D
    :param batch_norm Boolean flag to specify if a batch normalization is required
    :param activation Activation function as nn.Module
    :param bias Specify if bias is not null
    :param lift Specify if the latest layer is to be flattened
"""


class DeConvNeuralBlock(nn.Module):

    def __init__(self,
                 conv_dimension: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 batch_norm: bool,
                 activation: nn.Module,
                 bias: bool) -> object:
        super(DeConvNeuralBlock, self).__init__()
        self.conv_dimension = conv_dimension

        assert 0 < conv_dimension < 4, f'Conv neural block conv_dim {conv_dimension} should be {1, 2, 3}'
        assert in_channels > 0, f'Conv neural block disc_in_channels {in_channels} should be >0'
        assert out_channels > 0, f'Conv neural block out_channels {out_channels} should be >0'
        assert kernel_size > 0, f'Conv neural block kernel_size {kernel_size} should be >0'
        assert stride >= 0, f'Conv neural block stride {stride} should be >= 0'

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Need to add a lift for the first hidden layer
        self.modules = self.init1d(kernel_size,
                                   stride,
                                   padding,
                                   batch_norm,
                                   activation,
                                   bias) if conv_dimension == 1 \
            else self.init2d(kernel_size,
                             stride,
                             padding,
                             batch_norm,
                             activation,
                             bias)

    @classmethod
    def transpose_vae(cls,
                      conv_block: ConvNeuralBlock,
                      in_channels: int,
                      out_channels: int = None,
                      activation: nn.Module = None) -> nn.Module:
        """
            Build a de-convolutional neural block from a convolutional block
            :param conv_block: Convolutional neural block
            :param in_channels: Number of input_tensor channels used to override the value from convolutional model
                    if not None
            :param out_channels: Number of output channels adjusted
            :param activation: Activation which override the one found in the convolutional block if not None
            :return: Mirrored de-convolutional block
        """
        assert out_channels > 0, f'VAE Conv neural block out_channels {out_channels} should be >0'
        kernel_size, stride, padding, batch_norm, activation = DeConvNeuralBlock.__get_conv_params(conv_block,
                                                                                                   activation)
        next_block_out_channels = out_channels if out_channels is not None else in_channels // 2
        return cls(
            conv_block.conv_dimension,
            in_channels,
            next_block_out_channels,
            kernel_size,
            stride,
            padding,
            batch_norm,
            activation,
            False)

    @classmethod
    def transpose_gan(cls,
                      conv_block: ConvNeuralBlock,
                      in_channels: int,
                      out_channels: int = None,
                      activation: nn.Module = None) -> nn.Module:
        """"
            Build a de-convolutional neural block from a convolutional block
            :param conv_block: Convolutional neural block
            :param in_channels: Number of input_tensor channels used to override the value from convolution block
                if defined (not None)
            :param out_channels: Number of output channels used to override the current setting of the
                convolutional block if defined (not None)
            :param activation: Activation which override the one found in the convolutional block if not None
            :return: Mirrored de-convolutional block
        """
        kernel_size, stride, padding, batch_norm, activation = \
            DeConvNeuralBlock.__get_conv_params(conv_block, activation)
        # Override the number of input_tensor channels for this block if defined
        next_block_in_channels = in_channels if in_channels is not None \
            else conv_block.out_channels

        # OVerride the number of output-channels for this block if specifieed (not None)
        next_block_out_channels = out_channels if out_channels is not None \
            else conv_block.in_channels
        return cls(
            conv_block.conv_dimension,
            next_block_in_channels,
            next_block_out_channels,
            kernel_size,
            stride,
            padding,
            batch_norm,
            activation,
            False)

    def get_modules(self) -> list:
        return self.modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __repr__(self) -> str:
        return ' '.join([f'\n{str(module)}' for module in self.modules])

    # ---------------------------  Supporting methods -----------------------------
    @staticmethod
    def __get_conv_params(conv_block: ConvNeuralBlock,
                          updated_activation: nn.Module) -> (int, int, int, bool, nn.Module):
        conv_modules = list(conv_block.modules)
        # Extract the various components of the convolutional neural block
        _, batch_norm, activation = ConvModuleInstance.extract_conv_modules(conv_modules)
        # This override the activation function for the output layer, if necessary
        if updated_activation is not None:
            activation = updated_activation

        if conv_block.conv_dimension == 2:
            kernel_size, _ = conv_modules[0].kernel_size
            stride, _ = conv_modules[0].stride
            padding = conv_modules[0].padding
        else:
            kernel_size = conv_modules[0].kernel_size
            stride = conv_modules[0].stride
            padding = conv_modules[0].padding
        return kernel_size, stride, padding, batch_norm, activation

    def init1d(self,
               kernel_size: int,
               stride: int,
               padding: int,
               batch_norm: bool,
               activation: nn.Module,
               bias: bool) -> list:
        modules = []
        # Need to add a lift for the first hidden layer
        de_conv = nn.ConvTranspose1d(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        modules.append(de_conv)
        if batch_norm:
            modules.append(nn.BatchNorm1d(self.out_channels))
        modules.append(activation)
        return modules

    def init2d(self,
               kernel_size: int,
               stride: int,
               padding: int,
               batch_norm: bool,
               activation: nn.Module,
               bias: bool) -> list:
        modules = []
        # Need to add a lift for the first hidden layer
        de_conv = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        modules.append(de_conv)
        if batch_norm:
            modules.append(nn.BatchNorm2d(self.out_channels))
        modules.append(activation)
        return modules
