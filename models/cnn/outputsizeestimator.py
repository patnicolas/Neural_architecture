__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import math
from models.autoencoder.convaeconfig import ConvVAEBlockConfig

"""
    Estimate the size of output from a convolution, taking into account kernel size, stride, padding
    and pooling factor.
    This method applies equally to both 1D and 2D convolution. To compute the size of output of a layer, Nout given
    an input_tensor Nin, padding, kernel size, stride and dilation
    Convolution
        Nout = (Nin + 2.padding - dilation(kernel-1) - 1)/stride + 1
    Deconvolution
        Nout = stride.(Nin-1)+ 2.padding + dilation(kernel-1) + 1
    
    :param kernel_size: Size of the filtering kernel
    :type kernel_size: int
    :param stride: Stride for the convolutional filter
    :type stride: int
    :param padding: Padding used in resizing encoder after filter
    :type padding: int
    :param max_pooling_kernel_size: Optional scale factor for the max pooling
    :type max_pooling_kernel_size: int
"""


class OutputSizeEstimator(object):
    def __init__(self, kernel_size: int, stride: int, padding: int, max_pooling_kernel_size: int = None):
        assert kernel_size > 0, f'Kernel size {kernel_size} should be > 0'
        assert stride > 0, f'Stride {stride} should be > 0'
        assert padding > 0, f'Padding {padding} should be > 0'

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pooling_factor = max_pooling_kernel_size if max_pooling_kernel_size is not None else 1

    @staticmethod
    def get_output_size(input_size: int, block_config: ConvVAEBlockConfig, is_de_conv: bool) -> int:
        """
            Compute the size of the output for this convolutional block
            :param input_size: Input size
            :param block_config: Configuration for this block
            :param is_de_conv: Flag to specify, it is a de=convolutional block
            :return: output size
        """
        estimator = OutputSizeEstimator(block_config.kernel_size, block_config.stride, block_config.padding)
        return estimator.de_conv(input_size) if is_de_conv else estimator.conv(input_size)

    @staticmethod
    def get_output_sizes(input_size: int, block_configs: list, is_de_conv: bool) -> list:
        """
            Compute the size of output for a sequence of a convolutional or deconvolutional block
            :param input_size: Input size of the first block
            :param block_configs: List of configuration for convolutional blocks
            :param is_de_conv: Flag to specify, it is a de=convolutional block
            :return: List of output size
        """
        out_sizes = []
        in_size = input_size
        for block_config in block_configs:
            estimator = OutputSizeEstimator(block_config.kernel_size, block_config.stride, block_config.padding)
            if is_de_conv:
                out_size = estimator.de_conv(in_size)
            else:
                out_size = estimator.conv(in_size)
            out_sizes.append(out_size)
            in_size = out_size
        return out_sizes


    def conv(self, input_size: int, dilation: int = 1) -> int:
        """
            Compute the size of output from a convolutional neural block
                Nout = (Nin + 2.padding - dilation(kernel-1) - 1)/stride + 1
            :param input_size: Dimension of input_tensor encoder
            :param dilation Optional dilation value
            :returns: Dimension of output encoder
        """
        assert input_size > 0, f'Input size {input_size} should be > 0'
        assert dilation > 0, f'Dilation {dilation} should be > 0'

        return math.ceil((input_size + 2 * self.padding - self.kernel_size - 2) / self.stride) + 1 if dilation == 1 \
            else math.ceil((input_size + 2 * self.padding - dilation*(self.kernel_size - 1) -1) / self.stride) + 1


    def de_conv(self, input_size: int, dilation: int = 1) -> int:
        """
             Compute the size of output from a de-convolutional neural block
                Nout = stride.(Nin-1)+ 2.padding + dilation(kernel-1) + 1
             :param input_size: Dimension of input_tensor encoder
             :param dilation Optional dilation value
             :returns: Dimension of output encoder
         """
        assert input_size > 0, f'Input size {input_size} should be > 0'
        assert dilation > 0, f'Dilation {dilation} should be > 0'
        return self.stride*(input_size -1) - 2*self.padding + self.kernel_size+2 if dilation == 1 \
            else self.stride * (input_size - 1) - 2 * self.padding + dilation*(self.kernel_size + 1) + 1