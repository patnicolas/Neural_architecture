from unittest import TestCase

import torch
from models.cnn.deconvmodel import DeConvModel
from models.cnn.deconvnet import DeConvNet
from torch.utils.data import DataLoader
from models.cnn.deconvneuralblock import DeConvNeuralBlock
from models.nnet.hyperparams import HyperParams
import constants


class TestDeConvNet(TestCase):
    def test__train_and_eval(self):
        try:
            # Step 1: Create a convolutional encoder_model
            in_channels = 1
            hidden_dim = 16
            de_conv_model = TestDeConvNet.__create_de_conv_2d_model('DeConv2d', in_channels, hidden_dim)
            # Step 2: Define the hyper parameters
            batch_size = 20
            hyper_params = TestDeConvNet.__create_hyper_parameters(batch_size)


            # Step 3: Load data set
            data_loader = TestDeConvNet.load_data(batch_size)

            # Step 4: Train the encoder_model
            de_conv_net = DeConvNet(de_conv_model, hyper_params, None)
            # print(repr(de_conv_net))
            # input_tensor = TestDeConvNet.__input()
           # DeConvNet.forward(input_tensor, de_conv_model)
            de_conv_net.train_and_eval(data_loader.dataset)
        except Exception as e:
            self.fail(str(e))

    def test_build_2(self):
        conv_dimension = 2
        z_dim = 10
        hidden_dim = 14
        out_dim = 1
        params = [(3, 2, 1, True, torch.nn.LeakyReLU()), (2, 2, 0, True, torch.nn.LeakyReLU()),(2, 2, 0, False, torch.nn.Tanh())]
        de_conv_model = DeConvNet.feature_extractor('model-1', conv_dimension, z_dim, hidden_dim, out_dim, params)
        constants.log_info(repr(de_conv_model))
        batch_size = 20
        hyper_params = TestDeConvNet.__create_hyper_parameters(batch_size)

        # Step 3: Load data set
        data_loader = TestDeConvNet.load_data(batch_size)

        # Step 4: Train the encoder_model
        de_conv_net = DeConvNet(de_conv_model, hyper_params, None)
        de_conv_net.train_and_eval(data_loader.dataset)

    # ------------------  Supporting methods -------------------
    @staticmethod
    def __create_hyper_parameters(batch_size: int) -> HyperParams:
        lr = 0.001
        momentum = 0.99
        epochs = 20
        optim_label = 'adam'
        early_stop_patience = 3
        loss_func = torch.nn.BCEWithLogitsLoss()
        return HyperParams(lr, momentum, epochs, optim_label, batch_size, early_stop_patience, loss_func)


    @staticmethod
    def __create_de_conv_2d_model(model_id: str, in_channels: int, hidden_dim: int) -> DeConvModel:
        conv_dimension = 2
        de_conv_neural_block_1 = TestDeConvNet.__create_de_conv_2d_block(in_channels, hidden_dim * 4, torch.nn.ReLU(), True)
        de_conv_neural_block_2 = TestDeConvNet.__create_de_conv_2d_block(hidden_dim * 4, hidden_dim * 2, torch.nn.ReLU(), True)
        de_conv_neural_block_3 = TestDeConvNet.__create_de_conv_2d_block(hidden_dim * 2, hidden_dim, torch.nn.ReLU(), True)
        de_conv_neural_block_4 = TestDeConvNet.__create_de_conv_2d_block(hidden_dim, 10, torch.nn.Tanh(), False)
        return DeConvModel.build(model_id, [de_conv_neural_block_1, de_conv_neural_block_2, de_conv_neural_block_3, de_conv_neural_block_4])

    @staticmethod
    def __create_de_conv_2d_block(
            in_channels: int,
            out_channels: int,
            activation: torch.nn.Module,
            batch_norm: bool) -> DeConvNeuralBlock:
        kernel_size = 3
        bias = False
        stride = 2
        padding = 1
        conv_dimension = 2
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

    @staticmethod
    def load_data(batch_size: int) -> (DataLoader, DataLoader):
        from datasets.datasetloaders import DatasetLoaders
        data_loaders = DatasetLoaders(batch_size, -1, 0.85)
        return data_loaders.load_mnist([0.48, 0.52])

    @staticmethod
    def __input():
        return torch.tensor(
         [[[-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -0.8824, -0.6863,
          -0.6863, -0.4118, -0.2810, -0.1373,  0.8039,  0.6732, -0.6863,
          -0.6863, -0.9346, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -0.8301,
          -0.3856, -0.3856, -0.2549,  1.1046,  1.1046,  1.5359,  2.2941,
           2.3072,  2.2941,  2.2941,  2.2941,  1.7451,  2.3072,  2.2941,
           2.2941,  0.7647, -0.9346, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -0.5033,  0.8562,  1.6536,
           2.2941,  2.2941,  2.3072,  1.7190,  0.3072, -0.0980,  1.2614,
          -0.0980, -0.0980, -0.0980, -0.0980, -0.9216, -0.0980, -0.0980,
           0.5556,  2.2941,  0.3987, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000,  1.3529,  2.2941,  2.2941,
           1.3007,  1.0915,  0.2810, -0.7778, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -0.8301,  1.8105,  2.0327, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000,  2.3072,  2.2941,  0.1111,
          -0.9477, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000,  1.7059,  2.2941, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000,  0.6732,  2.3072,  1.9020,
          -0.6732, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -0.5033,  2.0458,  0.7908, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -0.8824,  1.2614,  2.2941,
          -0.4118, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -0.8824,
           1.0261,  2.2941, -0.1634, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -0.3856,  2.2941,
           0.6863, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -0.3856,
           2.2941,  1.0000, -0.9216, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -0.8824,  1.0131,
           1.5359, -0.8824, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -0.8824,  1.2745,
           1.9673, -0.6209, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -0.9085,
          -0.7124, -0.9739, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,  0.6732,  2.0196,
          -0.3725, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -0.8824,  0.6732,  2.2941,  0.7647,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -0.8301,  1.2745,  2.2941,  0.7647, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -0.4118,  1.2484,  2.2941,  0.7778, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000,  1.1046,  2.2941,  1.8497, -0.8170, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -0.0327,  2.1895,  1.4706, -0.8170, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
           2.3333,  2.1242, -0.1765, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -0.7124,
           2.3072,  0.2026, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,  0.5163,
           2.2288, -0.1242, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,  0.5163,
           2.0065, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
         [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -0.1765,
          -0.0458, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
          -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000]]])
