from unittest import TestCase
import torch
import constants
from models.cnn.deconvneuralblock import DeConvNeuralBlock
from models.cnn.convneuralblock import ConvNeuralBlock


class TestDeConvNeuralBlock(TestCase):
    def test_transpose_gan_no_channels(self):
        de_conv_block = DeConvNeuralBlock(conv_dimension=2,
                                          in_channels=16,
                                          out_channels=32,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          batch_norm=True,
                                          activation=torch.nn.ReLU(),
                                          bias=False)
        conv_block = ConvNeuralBlock.transpose_gan(de_conv_block, in_channels=None, activation=None)
        constants.log_info(repr(conv_block))


    def test_build(self):
        input_channels = 64
        output_channels = 32
        kernel_size = 3
        is_batch_normalization = True
        max_pooling_kernel = 2
        activation = torch.nn.Tanh()
        has_bias = False
        is_flatten = False
        stride = 2
        padding = 1,
        conv_block = ConvNeuralBlock(
            2,
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
        de_conv_block = DeConvNeuralBlock.transpose_gan(conv_block, 10, torch.nn.Sigmoid())
        constants.log_info(f'\nBuild conv_block: {repr(conv_block)}')
        constants.log_info(f'\nBuild de_conv_block {repr(de_conv_block)}')


    def test_get_modules_1d(self):
        try:
            de_conv_neural_block = TestDeConvNeuralBlock.__create_de_conv_block(1)
            print(repr(de_conv_neural_block))
            modules = de_conv_neural_block.get_modules()
            modules_str = '\n'.join([str(module) for module in modules])
            constants.log_info(f'Get modules 1D:\n{modules_str}')
        except Exception as e:
            self.fail(str(e))

    def test_get_modules_2d(self):
        try:
            de_conv_neural_block = TestDeConvNeuralBlock.__create_de_conv_block(2)
            print(repr(de_conv_neural_block))
            modules = de_conv_neural_block.get_modules()
            modules_str = '\n'.join([str(module) for module in modules])
            constants.log_info(f'Get modules 2D:\n{modules_str}')
        except Exception as e:
            self.fail(str(e))

    @staticmethod
    def __create_de_conv_block(conv_dimension: int) -> DeConvNeuralBlock:
        in_channels = 64
        out_channels = 32
        kernel_size = 3
        batch_norm = True
        activation = torch.nn.Tanh()
        bias = False
        stride = 1
        padding = 1
        return DeConvNeuralBlock(
                    conv_dimension,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    batch_norm,
                    activation,
                    bias)
