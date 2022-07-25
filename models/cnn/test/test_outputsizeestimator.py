from unittest import TestCase

from models.cnn.outputsizeestimator import OutputSizeEstimator
import math
import torch
import constants
from models.autoencoder.convaeconfig import ConvVAEBlockConfig


class TestOutputSizeEstimator(TestCase):
    def test_conv(self):
        try:
            kernel_size = 4
            padding = 2
            stride = 2
            output_size_estimator = OutputSizeEstimator(kernel_size, stride, padding, None)
            input_size = 11
            est = (input_size + 2*output_size_estimator.padding - output_size_estimator.kernel_size)/output_size_estimator.stride
            expected = math.ceil(est) + 1
            self.assertEqual(output_size_estimator.conv(input_size), expected)
        except Exception as e:
            self.fail(str(e))


    def test_de_conv(self):
        try:
            kernel_size = 4
            padding = 2
            stride = 2
            output_size_estimator = OutputSizeEstimator(kernel_size, stride, padding, None)
            input_size = 11
            expected = output_size_estimator.stride*(input_size - 1) - 2*output_size_estimator.padding + output_size_estimator.kernel_size
            self.assertEqual(output_size_estimator.de_conv(input_size), expected)
        except Exception as e:
            self.fail(str(e))


    def test_get_output_size(self):
        conv_configs = [
            ConvVAEBlockConfig(2, 4, 2, 1, 1, torch.nn.LeakyReLU(0.2), False),
            ConvVAEBlockConfig(2, 4, 2, 2, 1, torch.nn.LeakyReLU(0.2), False),
            ConvVAEBlockConfig(2, 4, 2, 1, 2, torch.nn.LeakyReLU(0.2), False)
        ]
        output_size = OutputSizeEstimator.get_output_size(12, conv_configs[0], False)
        constants.log_info(f'Conv output_size: {output_size}')

        output_sizes = OutputSizeEstimator.get_output_sizes(12, conv_configs, False)
        constants.log_info(f'Conv output_sizes: {str(output_sizes)}')

        output_size_2 = OutputSizeEstimator.get_output_size(12, conv_configs[0], True)
        constants.log_info(f'Deconv output_size: {output_size_2}')

        output_sizes_2 = OutputSizeEstimator.get_output_sizes(12, conv_configs, True)
        constants.log_info(f'Deconv output_sizes: {str(output_sizes_2)}')



