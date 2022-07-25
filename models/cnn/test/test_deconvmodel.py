from unittest import TestCase

import torch
import unittest
import constants
from models.cnn.deconvmodel import DeConvModel
from models.cnn.deconvneuralblock import DeConvNeuralBlock
from models.cnn.convmodel import ConvModel
from models.cnn.convneuralblock import ConvNeuralBlock


class TestDeConvModel(TestCase):
    def test_save(self):
        conv_dimension = 2
        input_channels = 1
        output_channels = 1
        hidden_layer_size = 4
        activation = torch.nn.ReLU()
        output_size = 2
        num_blocks = 1
        de_conv_model = TestDeConvModel.__create_de_conv_model(
            num_blocks,
            conv_dimension,
            input_channels,
            output_channels,
            activation=torch.nn.Tanh(),
            hidden_layer_size=hidden_layer_size,
            connected_activation=activation,
            output_size=output_size)

        de_conv_model.save(None)
        de_conv_model2 = DeConvModel.load("deconvtest1")
        constants.log_info(repr(de_conv_model2))

    @unittest.skip("Not needed")
    def test_build(self):
        conv_dimension = 2
        block1 = ConvNeuralBlock(
            conv_dimension,
            10,
            32,
            4,
            2,
            1,
            True,
            2,
            torch.nn.ReLU(),
            False,
            False)
        block2 = ConvNeuralBlock(
            conv_dimension,
            32,
            64,
            2,
            1,
            1,
            True,
            2,
            torch.nn.ReLU(),
            False,
            False)
        block3 = ConvNeuralBlock(
            conv_dimension,
            64,
            128,
            3,
            2,
            1,
            True,
            1,
            torch.nn.Tanh(),
            False,
            False)

        # kernel_size, stride, padding, is_batch_norm, max_pooling_kernel, activation
        conv_blocks = [block1, block2, block3]
        conv_model = ConvModel.build("conv_model", 2, conv_blocks, None)
        constants.log_info(f'\nConv model: {repr(conv_model)}')
        de_conv_model = DeConvModel.build_for_gan("de_conv_model", conv_blocks, 10, None, torch.nn.Sigmoid())
        constants.log_info(f'\nDe conv model: {repr(de_conv_model)}')

    @unittest.skip("Not needed")
    def test_forward_1d_in1d_not_connected(self):
        try:
            conv_dimension = 1
            input_channels = 1
            output_channels = 1
            num_blocks = 2
            conv_model = TestDeConvModel.__create_de_conv_model(
                num_blocks,
                conv_dimension,
                input_channels,
                output_channels,
                activation=torch.nn.Tanh())
            print(repr(conv_model))
            x = torch.arange(0.0, 12.8, 0.1)
            y = torch.sin(x).unsqueeze(0).unsqueeze(0)
            print(y)
            output = conv_model(y)
            print(output)
        except Exception as e:
            self.fail(str(e))

    @unittest.skip('Not needed')
    def test_forward_1d_in1d_connected(self):
        try:
            conv_dimension = 1
            input_channels = 1
            output_channels = 1
            hidden_layer_size = 128
            activation = torch.nn.Sigmoid()
            output_size: int = 2
            num_blocks = 2
            conv_model = TestDeConvModel.__create_de_conv_model(
                num_blocks,
                conv_dimension,
                input_channels,
                output_channels,
                activation=torch.nn.Tanh(),
                hidden_layer_size=hidden_layer_size,
                connected_activation=activation,
                output_size=output_size)

            print(repr(conv_model))
            x = torch.arange(0.0, 12.8, 0.1)
            y = torch.sin(x).unsqueeze(0).unsqueeze(0)
            print(y)
            output = conv_model(y)
            print(output)
        except Exception as e:
            self.fail(str(e))

    @unittest.skip('Not needed')
    def test_forward_1d_in2d_connected(self):
        try:
            conv_dimension = 1
            input_channels = 2
            output_channels = 1
            hidden_layer_size = 4
            activation = torch.nn.Sigmoid()
            output_size: int = 2
            num_blocks = 1
            deconv_model = TestDeConvModel.__create_de_conv_model(
                num_blocks,
                conv_dimension,
                input_channels,
                output_channels,
                activation=torch.nn.Tanh(),
                hidden_layer_size=hidden_layer_size,
                connected_activation=activation,
                output_size=output_size)
            print(repr(deconv_model))
            y = torch.tensor([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=torch.float)
            y = y.unsqueeze(0)
            print(list(y.size()))
            output = deconv_model(y)
            print(output)
        except Exception as e:
            self.fail(str(e))

    @unittest.skip('Not needed')
    def test_forward_2d_in2d_connected(self):
        try:
            conv_dimension = 2
            input_channels = 1
            output_channels = 1
            hidden_layer_size = 4
            activation = torch.nn.ReLU()
            output_size = 2
            num_blocks = 1
            conv_model = TestDeConvModel.__create_de_conv_model(
                num_blocks,
                conv_dimension,
                input_channels,
                output_channels,
                activation=torch.nn.Tanh(),
                hidden_layer_size=hidden_layer_size,
                connected_activation=activation,
                output_size=output_size)
            print(repr(conv_model))

            y = torch.tensor([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=torch.float)
            y = y.unsqueeze(0)
            print(list(y.size()))
            print(y)
            output = conv_model(y)
            print(output)
        except Exception as e:
            self.fail(str(e))

            # ----------------- Supporting methods -------------
    @staticmethod
    def __create_de_conv_model(
            num_blocks: int,
            conv_dimension: int,
            input_channels: int,
            output_channels: int,
            activation: torch.nn.Module,
            hidden_layer_size: int = -1,
            connected_activation: torch.nn.Module = None,
            output_size: int = -1) -> DeConvModel:
        conv_neural_block_1 = TestDeConvModel.__create_de_conv_block(conv_dimension, input_channels, output_channels, activation)
        if num_blocks > 1:
            conv_neural_block_2 = TestDeConvModel.__create_de_conv_block(conv_dimension, input_channels, output_channels, activation)
            blocks = [conv_neural_block_1, conv_neural_block_2]
        else:
            blocks = [conv_neural_block_1]
        return DeConvModel.build_for_gan("deconvtest1", blocks, hidden_layer_size, output_size, connected_activation)


    @staticmethod
    def __create_de_conv_block(conv_dimension: int,
                               input_channels: int,
                               output_channels: int,
                               activation: torch.nn.Module) -> DeConvNeuralBlock:
        kernel_size = 1
        is_batch_normalization = True
        is_max_pooling = False
        has_bias = False
        is_flatten = False
        stride = 1
        padding = 0
        return DeConvNeuralBlock(
            conv_dimension,
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            is_batch_normalization,
            activation,
            has_bias)

    @staticmethod
    def __create_conv_model(
            conv_dimension: int,
            input_channels: int,
            output_channels: int,
            activation: torch.nn.Module,
            dff_blocks: list = None) -> ConvModel:

        conv_neural_block_1 = TestDeConvModel.__create_conv_block(conv_dimension, input_channels, input_channels * 2, activation)
        conv_neural_block_2 = TestDeConvModel.__create_conv_block(conv_dimension, input_channels * 2, output_channels, activation)
        blocks = [conv_neural_block_1, conv_neural_block_2]
        return ConvModel.build("test1-1", conv_dimension, blocks, dff_blocks)

    @staticmethod
    def __create_conv_block(dim: int, input_channels: int, output_channels: int, activation: torch.nn.Module) -> ConvNeuralBlock:
        kernel_size = 2
        is_batch_normalization = True
        max_pooling_kernel = 2
        has_bias = False
        padding = 1
        is_flatten = False
        stride = 1
        return ConvNeuralBlock(
            dim,
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