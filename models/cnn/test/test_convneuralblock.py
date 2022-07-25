from unittest import TestCase
import unittest
import torch
from models.cnn.convneuralblock import ConvNeuralBlock


class TestConvNeuralBlock(TestCase):

    @unittest.skip("Not needed")
    def test_get_modules_1d(self):
        try:
            conv_dimension = 1
            conv_neural_block = TestConvNeuralBlock.__create_conv_block(conv_dimension)
            print(repr(conv_neural_block))
            modules = conv_neural_block.get_modules()
            print('\n'.join([str(module) for module in modules]))
        except Exception as e:
            self.fail(str(e))


    @unittest.skip("Not needed")
    def test_get_modules_2d(self):
        try:
            conv_dimension = 2
            conv_neural_block = TestConvNeuralBlock.__create_conv_block(conv_dimension)
            print(repr(conv_neural_block))
            modules = conv_neural_block.get_modules()
            print('\n'.join([str(module) for module in modules]))
        except Exception as e:
            self.fail(str(e))

    @staticmethod
    def __create_conv_block(conv_dimension: int) -> ConvNeuralBlock:
        input_channels = 64
        output_channels = 32
        kernel_size = 3
        is_batch_normalization = True
        max_pooling_kernel = 2
        activation = torch.nn.Tanh()
        has_bias = False
        is_flatten = False
        stride = 1
        padding = 1,
        return ConvNeuralBlock(
            conv_dimension,
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            is_batch_normalization,
            max_pooling_kernel,
            activation,
            has_bias,
            is_flatten)