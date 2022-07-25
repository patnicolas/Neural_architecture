import torch
import constants
from torch.utils.data import DataLoader
from models.cnn import ConvNeuralBlock
from models.cnn import DeConvNeuralBlock
from models.cnn.convmodel import ConvModel
from models.cnn import DeConvModel
from models.nnet.hyperparams import HyperParams
from models.gan.dcgenerator import DCGenerator
from models.cnn.snneuralblock import SNNeuralBlock
from models.gan.dcdiscriminator import DCDiscriminator
from util.imagetensor import ImageTensor

"""
    Builder for MNIST-based convolutional and de-convolutional neural models used in various GAN such
    as Deep convolutional GAN, Wasserstein GAN, Conditional convolutional GAN,....
    All methods are static
"""


class MNISTDriver(object):

    @staticmethod
    def debug(image_tensor: torch.Tensor, num_images: int):
        if constants.is_log_debug:
            ImageTensor.show_tensor_images(image_tensor, num_images, 4)

    @staticmethod
    def create_conv_model_auto(conv_dimension: int,
                               in_channels: int = 1,
                               out_channels: int = 1,
                               hidden_dim: int = 16) -> list:
        kernel_size = 4
        stride = 2
        padding = 1
        batch_normalization = True
        max_pooling_kernel = 0
        neural_blocks = []
        for idx in range(5):
            if idx == 0:
                in_chans = in_channels
                out_chans = hidden_dim
            elif idx == 4:
                in_chans = out_chans
                out_chans = out_channels
            else:
                in_chans = out_chans
                out_chans = in_chans * 2
            neural_blocks.append(ConvNeuralBlock(conv_dimension,
                                                 in_chans,
                                                 out_chans,
                                                 kernel_size,
                                                 stride,
                                                 padding,
                                                 batch_normalization,
                                                 max_pooling_kernel,
                                                 torch.nn.LeakyReLU(0.2),
                                                 False,
                                                 False))
        return neural_blocks


    @staticmethod
    def create_sn_conv_model_auto(conv_dimension: int,
                                  in_channels: int = 1,
                                  out_channels: int = 1,
                                  hidden_dim: int = 16) -> list:
        kernel_size = 4
        stride = 2
        padding = 1
        batch_normalization = True
        max_pooling_kernel = 0
        neural_blocks = []
        for idx in range(5):
            if idx == 0:
                in_chans = in_channels
                out_chans = hidden_dim
            elif idx == 4:
                in_chans = out_chans
                out_chans = out_channels
            else:
                in_chans = out_chans
                out_chans = in_chans * 2
            sn_block = SNNeuralBlock(conv_dimension,
                                     in_chans,
                                     out_chans,
                                     kernel_size,
                                     stride,
                                     padding,
                                     batch_normalization,
                                     max_pooling_kernel,
                                     torch.nn.LeakyReLU(0.2))
            neural_blocks.append(sn_block)
        return neural_blocks


    @staticmethod
    def create_conv_model(conv_dimension: int, in_channels: int = 1, out_channels: int = 1) -> list:
        kernel_size = 4
        stride = 2
        padding = 1
        batch_normalization = True
        max_pooling_kernel = 0
        block1 = ConvNeuralBlock(
            conv_dimension,
            in_channels,
            16,
            kernel_size,
            stride,
            padding,
            batch_normalization,
            max_pooling_kernel,
            torch.nn.LeakyReLU(0.2),
            False,
            False)
        block2 = ConvNeuralBlock(
            conv_dimension,
            16,
            32,
            kernel_size,
            stride,
            padding,
            batch_normalization,
            max_pooling_kernel,
            torch.nn.LeakyReLU(0.2),
            False,
            False)
        block3 = ConvNeuralBlock(
            conv_dimension,
            32,
            64,
            kernel_size,
            stride,
            padding,
            batch_normalization,
            max_pooling_kernel,
            torch.nn.LeakyReLU(0.2),
            False,
            False)
        block4 = ConvNeuralBlock(
            conv_dimension,
            64,
            128,
            kernel_size,
            stride,
            padding,
            batch_normalization,
            max_pooling_kernel,
            torch.nn.LeakyReLU(0.2),
            False,
            False)
        block5 = ConvNeuralBlock(
            conv_dimension,
            128,
            out_channels,
            kernel_size,
            stride,
            padding,
            batch_normalization,
            max_pooling_kernel,
            torch.nn.LeakyReLU(0.2),
            False,
            False)
        return [block1, block2, block3, block4, block5]

    @staticmethod
    def create_2d_conv_model(model_id: str, input_dim: int, hidden_dim: int, output_dim: int) -> ConvModel:
        conv_neural_block_1 = MNISTDriver.__create_2d_conv_block(
            2,
            input_dim,
            hidden_dim,
            torch.nn.LeakyReLU(0.2),
            True,
            0)
        # nout = (22 + 2*0 - 4)/2 + 1 = 10
        conv_neural_block_2 = MNISTDriver.__create_2d_conv_block(
            2,
            hidden_dim,
            hidden_dim * 2,
            torch.nn.LeakyReLU(0.2),
            True,
            0)
        # nout = (10 + 2*0 -4)/2 + 1 = 4
        conv_neural_block_3 = MNISTDriver.__create_2d_conv_block(
            2,
            hidden_dim * 2,
            output_dim,
            torch.nn.LeakyReLU(0.2),
            True,
            0)
        # nout = (4 + 2*0 - 4)/2 + 1 = 1
        return ConvModel.build(
            model_id,
            2,
            [conv_neural_block_1, conv_neural_block_2, conv_neural_block_3],
            None)


    @staticmethod
    def __create_2d_conv_block(
            dim: int,
            input_dim: int,
            output_dim: int,
            activation: torch.nn.Module,
            batch_norm: bool,
            padding: int) -> ConvNeuralBlock:
        kernel_size = 4
        max_pooling_kernel = 0
        bias = False
        flatten = False
        stride = 2
        return ConvNeuralBlock(
            dim,
            input_dim,
            output_dim,
            kernel_size,
            stride,
            padding,
            batch_norm,
            max_pooling_kernel,
            activation,
            bias,
            flatten)


    @staticmethod
    def create_de_2d_conv_model(model_id: str, input_dim: int, hidden_dim: int, output_dim: int) -> DeConvModel:
        de_conv_neural_block_1 = MNISTDriver.__create_2d_de_conv_block(
            2,
            input_dim,
            hidden_dim * 2,
            torch.nn.LeakyReLU(0.2),
            True,
            2,
            0)
        # nout = 2(1 -1) -2*0 +4 = 4
        de_conv_neural_block_2 = MNISTDriver.__create_2d_de_conv_block(
            2,
            hidden_dim * 2,
            hidden_dim,
            torch.nn.LeakyReLU(0.2),
            True,
            # 2,
            3,
            0)
        # nout = 2(4-1) - 0 + 4 = 10
        de_conv_neural_block_3 = MNISTDriver.__create_2d_de_conv_block(
            2,
            hidden_dim,
            output_dim,
            torch.nn.Sigmoid(),
            True,
            2,
            0)
        # nout = 2(10-1) + 4 = 22
        return DeConvModel.build(model_id, [de_conv_neural_block_1, de_conv_neural_block_2, de_conv_neural_block_3])


    @staticmethod
    def __create_2d_de_conv_block(
            dim: int,
            input_dim: int,
            output_dim: int,
            activation: torch.nn.Module,
            batch_norm: bool,
            stride: int,
            padding: int) -> DeConvNeuralBlock:
        kernel_size = 4
        bias = False
        return DeConvNeuralBlock(
            dim,
            input_dim,
            output_dim,
            kernel_size,
            stride,
            padding,
            batch_norm,
            activation,
            bias)


    @staticmethod
    def load_data(batch_size: int, num_samples: int) -> (DataLoader, DataLoader):
        from datasets.datasetloaders import DatasetLoaders
        data_loaders = DatasetLoaders(batch_size, num_samples, 0.85)
        return data_loaders.load_mnist([0.48, 0.52])


    @staticmethod
    def create_hyper_params_adam(lr: float,
                                 epochs: int,
                                 batch_size: int,
                                 loss_function: torch.nn.Module) -> HyperParams:
        momentum = 0.9
        optim_label = 'adam'
        early_stop_patience = 3
        return HyperParams(lr, momentum, epochs, optim_label, batch_size, early_stop_patience, loss_function)


    @staticmethod
    def discriminator_and_generator(disc_id: str,
                                    gen_id: str,
                                    in_channels: int,
                                    hidden_dim: int,
                                    out_channels: int,
                                    z_dim: int,
                                    unsqueeze: bool = False) -> (DCDiscriminator, DCGenerator):
        conv_2d_model = MNISTDriver.create_2d_conv_model(disc_id, in_channels, hidden_dim, out_channels)
        dc_discriminator = DCDiscriminator(conv_2d_model)

        de_conv_2d_model = MNISTDriver.create_de_2d_conv_model(gen_id, z_dim, hidden_dim, out_channels)
        dc_generator = DCGenerator(de_conv_2d_model, z_dim, unsqueeze)
        return dc_discriminator, dc_generator

"""
 conv_dimension = 2
            output_channels = 12
            output_size = 4
            activation = torch.nn.Sigmoid()
"""