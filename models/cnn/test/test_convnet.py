import unittest
from unittest import TestCase

import torch
from models.cnn.convmodel import ConvModel
from models.cnn.convneuralblock import ConvNeuralBlock
from torch.utils.data import DataLoader
from models.nnet.hyperparams import HyperParams
from models.cnn.convnet import ConvNet
from models.dffn.dffneuralblock import DFFNeuralBlock
import constants


class TestConvNet(TestCase):
    @unittest.skip("Not needed")
    def test_load_data(self):
        try:
            batch_size = 18
            data_loader = TestConvNet.load_data(batch_size)
            constants.log_info(f'Size data set {len(data_loader)}')
            first_entry = iter(data_loader)
            constants.log_info(str(next(first_entry)))
        except Exception as e:
            self.fail(str(e))


    @unittest.skip("Not needed")
    def test_train_and_eval(self):
        try:
            # Step 1: Create a convolutional encoder_model
            in_channels = 1
            hidden_dim = 12
            output_size = 10
            conv_model = TestConvNet.__create_2d_model('Conv2d', in_channels, hidden_dim, output_size)

            # Step 2: Define the hyper parameters
            batch_size = 18
            hyper_params = TestConvNet.__create_hyper_parameters(batch_size)

            # Step 3: Load data set
            train_loader, valid_loader = TestConvNet.load_data(batch_size)

            # Step 4: Train the encoder_model
            conv_net = ConvNet(conv_model, hyper_params, None)
            constants.log_info(repr(conv_net))
            conv_net.train_then_eval(train_loader, valid_loader)
        except Exception as e:
            self.fail(str(e))


    def test_build(self):
        conv_dimension = 2
        in_channels = 1
        hidden_dim = 16
        out_dim = 10
        # kernel_size, stride, padding, is_batch_norm, max_pooling_kernel, activation
        params = [
            (3, 1, 1, True, 2, torch.nn.LeakyReLU()),
            (2, 1, 0, True, 2, torch.nn.LeakyReLU()),
            (2, 1, 0, False, 1, torch.nn.Tanh())
        ]
        # Cascading model
        conv_model = ConvNet.feature_extractor('model-1', conv_dimension, in_channels, hidden_dim, out_dim, params)
        constants.log_info(repr(conv_model))
        batch_size = 18
        hyper_params = TestConvNet.__create_hyper_parameters(batch_size)

        # Step 3: Load data set
        train_loader, valid_loader = TestConvNet.load_data(batch_size)

        # Step 4: Train the encoder_model
        conv_net = ConvNet(conv_model, hyper_params, None)
        conv_net.train_then_eval(train_loader, valid_loader)

    # --------------------------  Supporting methods -----------------------------------------

    @staticmethod
    def loss_function(predicted: torch.Tensor, labels: torch.Tensor) -> float:
        from torch.autograd import Variable
        return torch.nn.NLLLoss()(Variable(predicted.float().data, requires_grad=True), labels)

    @staticmethod
    def __create_hyper_parameters(batch_size: int) -> HyperParams:
        lr = 0.001
        momentum = 0.95
        epochs = 10
        optim_label = 'adam'
        early_stop_patience = 3
        loss_func = torch.nn.NLLLoss()
        return HyperParams(lr, momentum, epochs, optim_label, batch_size, early_stop_patience, loss_func)

    @staticmethod
    def __create_2d_model(model_id: str, in_channels: int, hidden_dim: int, output_size: int) -> ConvModel:
        conv_neural_block_1 = TestConvNet.__create_2d_block(2, in_channels, hidden_dim, torch.nn.LeakyReLU(0.2), False)
        conv_neural_block_2 = TestConvNet.__create_2d_block(2, hidden_dim, hidden_dim * 2, torch.nn.LeakyReLU(0.2), False)
        dff_neural_block_3 = DFFNeuralBlock(hidden_dim*2, hidden_dim, torch.nn.ReLU(), 0.2)
        dff_neural_block_4 = DFFNeuralBlock(hidden_dim, output_size, torch.nn.LogSoftmax(dim=1), -1.0)
        return ConvModel.build(model_id, [conv_neural_block_1, conv_neural_block_2], [dff_neural_block_3, dff_neural_block_4])

    @staticmethod
    def __create_2d_block(
            conv_dimension: int,
            in_channels: int,
            out_channels: int,
            activation: torch.nn.Module,
            batch_norm: bool) -> ConvNeuralBlock:
        kernel_size = 4
        max_pooling_kernel = 2
        bias = False
        flatten = False
        stride = 2
        padding = 1,
        return ConvNeuralBlock(
            conv_dimension,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            batch_norm,
            max_pooling_kernel,
            activation,
            bias,
            flatten)

    @staticmethod
    def load_data(batch_size: int) -> (DataLoader, DataLoader):
        """
            Generate a training and validation data loader for MNIST images
            :param batch_size: Size of the batch size
            :return: Tuple training_loader, evaluation_loader
        """
        from datasets.datasetloaders import DatasetLoaders

        data_loaders = DatasetLoaders(batch_size, -1, 0.85)
        return data_loaders.load_mnist([0.48, 0.52])
