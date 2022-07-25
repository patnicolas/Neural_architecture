from unittest import TestCase

import torch
import unittest
from models.cnn.convneuralblock import ConvNeuralBlock
from models.cnn.convmodel import ConvModel
from models.dffn.dffneuralblock import DFFNeuralBlock
from models.nnet.neuralmodel import NeuralModel
import constants


class TestConvModel(TestCase):
    def test_save(self):
        input_channels = 1
        conv_dimension = 2
        output_channels = 12
        conv_model = TestConvModel.__create_conv_model(
            conv_dimension,
            input_channels,
            output_channels,
            activation=torch.nn.Tanh())
        conv_model.save(None)
        conv_net = ConvModel.load("convtest1")
        constants.log_info(repr(conv_net))

    @unittest.skip("Not needed")
    def test_forward_1d_in1d_not_connected(self):
        try:
            conv_dimension = 1
            input_channels = 1
            output_channels = 1
            conv_model = TestConvModel.__create_conv_model(
                conv_dimension,
                input_channels,
                output_channels,
                activation = torch.nn.Tanh())
            print(repr(conv_model))
            x = torch.arange(0.0, 12.8, 0.1)
            y = torch.sin(x).unsqueeze(0).unsqueeze(0)
            constants.log_size(y, 'input_tensor y')
            print(y)
            output = conv_model(y)
            print(output)
        except Exception as e:
            self.fail(str(e))

    @unittest.skip("Not needed")
    def test_forward_1d_in1d_connected(self):
        try:
            x = torch.arange(0.0, 12.8, 0.1)
            constants.log_size(x, 'actual')
            conv_dimension = 1

            # The output channel of the last convolution should be the input_tensor channel of the fully
            # connected layer
            output_channels = 2
            output_size = 10
            activation = torch.nn.Sigmoid()
            drop_out_factor = 0.2
            dff_blocks = [DFFNeuralBlock(output_channels, output_size, activation, drop_out_factor)]

            # First input_tensor channel (1 feature time series) and output two channel
            input_channels = 1
            conv_model = TestConvModel.__create_conv_model(
                conv_dimension,
                input_channels,
                output_channels,
                torch.nn.Tanh(),
                dff_blocks)

            print(repr(conv_model))
            y = torch.sin(x).unsqueeze(0).unsqueeze(0)
            constants.log_size(y, 'y sin')
            output = conv_model(y)
            print(output.detach())
        except Exception as e:
            self.fail(str(e))

    @unittest.skip("Not needed")
    def test_forward_2d_in2d_connected(self):
        try:
            conv_dimension = 2
            input_channels = 1
            output_channels = 2
            conv_model = TestConvModel.__create_conv_model(
                conv_dimension,
                input_channels,
                output_channels,
                activation=torch.nn.Tanh())
            constants.log_info(repr(conv_model))
            # Simulate input_tensor with 1 channel (1 feature)
            y = torch.tensor([[[1, 2, 3, 4], [6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17]]], dtype=torch.float)
            constants.log_size(y, 'y')

            # Needed to create a batch of 1 doc_terms_str
            y = y.unsqueeze(0)
            constants.log_size(y, '_y')
            output = conv_model(y)
            constants.log_info(output.detach())
        except Exception as e:
            self.fail(str(e))

    @unittest.skip("Not needed")
    def test_forward_2d_in2d_connected_2(self):
        try:
            conv_dimension = 2
            output_channels = 12
            output_size = 4
            activation = torch.nn.Sigmoid()
            drop_out_factor = 0.2
            dff_blocks = [DFFNeuralBlock(output_channels, output_size, activation, drop_out_factor)]

            input_channels = 2
            conv_model = TestConvModel.__create_conv_model(
                conv_dimension,
                input_channels,
                output_channels,
                torch.nn.ReLU(),
                dff_blocks)
            constants.log_info(repr(conv_model))
            # Simulate an image with two channels
            y = torch.tensor([[[1, 2, 3, 4], [6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17]],
                              [[11, 12, 13, 14], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27]]], dtype=torch.float)
            constants.log_size(y, 'y')
            # Needed to create a batch of 1 doc_terms_str
            y = y.unsqueeze(0)
            constants.get_logger().size(y, 'y_')
            output = conv_model(y)
            constants.log_info(output.detach())
        except Exception as e:
            self.fail(str(e))


    @staticmethod
    def __create_conv_model(
            conv_dimension: int,
            input_channels: int,
            output_channels: int,
            activation: torch.nn.Module,
            dff_blocks: list = None) -> NeuralModel:

        conv_neural_block_1 = TestConvModel.__create_conv_block(conv_dimension, input_channels, input_channels * 2, activation)
        conv_neural_block_2 = TestConvModel.__create_conv_block(conv_dimension, input_channels * 2, output_channels, activation)
        blocks = [conv_neural_block_1, conv_neural_block_2]
        return ConvModel.build("convtest1", conv_dimension, blocks, dff_blocks, False)

    @staticmethod
    def __create_conv_block(dim: int, input_channels: int, output_channels: int, activation: torch.nn.Module) -> ConvNeuralBlock:
        kernel_size = 2
        is_batch_normalization = True
        max_pooling_kernel = 2
        has_bias = False
        stride = 1
        return ConvNeuralBlock(
            dim,
            input_channels,
            output_channels,
            kernel_size,
            stride,
            1,
            is_batch_normalization,
            max_pooling_kernel,
            activation,
            has_bias,
            is_spectral = False)