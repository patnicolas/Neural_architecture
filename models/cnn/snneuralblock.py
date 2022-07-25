__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from models.cnn.convneuralblock import ConvNeuralBlock

"""    
    Spectral norm convolutional neural block for 1 and 2 dimensions. It is basically a convolutional neural block
    with SN-spectral flag = True
    Components:
         Spectral_convolution
         Batch normalization (Optional)
         Activation
         Max pooling (Optional)

    :param conv_dim Dimension of the convolution (1 or 2)
    :param in_channels Number of input_tensor channels
    :param output_dim Number of output channels
    :param kernel_size Size of the kernel (num_records) for 1D and (num_records, num_records) for 2D
    :param stride Stride for convolution (st) for 1D, (st, st) for 2D
    :param batch_norm Boolean flag to specify if a batch normalization is required
    :param max_pooling_kernel Boolean flag to specify max pooling is neede
    :param activation Activation function as nn.Module
"""


class SNNeuralBlock(ConvNeuralBlock):
    def __init__(self,
                 conv_dimension: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 batch_norm: bool,
                 max_pooling_kernel: int,
                 activation: torch.nn.Module):
        super(SNNeuralBlock, self).__init__(conv_dimension,
                                            in_channels,
                                            out_channels,
                                            kernel_size,
                                            stride,
                                            padding,
                                            batch_norm,
                                            max_pooling_kernel,
                                            activation,
                                            bias=False,
                                            is_spectral=True)
