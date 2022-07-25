__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from util.plottermixin import PlotterMixin
from models.nnet.neuralnet import NeuralNet
from models.cnn.convmodel import ConvModel
from models.cnn.convneuralblock import ConvNeuralBlock
from models.nnet.hyperparams import HyperParams
from torch.utils.data import Dataset, DataLoader


"""
    Configurable convolutional neural network. 
    - A convolutional neural network is composed of neural block
    - Each block is composed of Convolution, Batch normalization, activation, max pooling and optionally a 
    dropout module
    This class inherits from 
    - NeuralNet for training and evaluation
    - PlotterMixin for displaying 
    
    :param conv_vae_model: Convolutional Neural Network encoder_model or layout as a composition of Neural blocks
    :type conv_vae_model: cnn.convnetmodel.ConvNetModel
    :param hyper_params: Training (hyper) parameters used for tuning
    :type hyper_params: nnet.hyperparams.HyperParams
    :param debug: Transform the label input_tensor prior to loss function (encoder, labels) -> converted labels
    :type debug: (list[torch.Tensor], list[torch.Tensor]) -> torch.Tensor
    :param post_epoch_func:
"""


class ConvNet(NeuralNet, PlotterMixin):
    def __init__(self,
                 conv_net_model: ConvModel,
                 hyper_params: HyperParams,
                 debug):
        super(ConvNet, self).__init__(hyper_params, debug)
        self.conv_net_model = conv_net_model

    @classmethod
    def init(cls, config, conv_net_model: ConvModel, loss_func: torch.nn.Module, debug):
        """
            Alternative constructor for hyper-parameters tuning
            :param config: Ray tune config
            :param conv_net_model: Convolutional model
            :param loss_func: Loss function
            :param debug:
            :returns: instance of Neural net
        """
        hyper_params = HyperParams(config['learning-rate'], config['momentum'], 30, config['batch_size'], 10.0, loss_func)
        return ConvNet(conv_net_model, hyper_params, debug)

    def apply_debug(self, features: list, labels: list, title: str):
        """
            Apply a debug information related to the list of encoder (for debugging purpose)
            :param features: list of encoder
            :param labels: List of labels
            :param title: Title or description of the debugging info
        """
        if self._debug is not None:
            self._debug(features, labels, title)

    def model_label(self) -> str:
        return self.conv_net_model.model_label()

    def train_and_eval(self, dataset: Dataset):
        """
            Training and evaluation of a given encoder_model, conv_vae_model, with a given set of the hyper-parameters
            :param dataset: Data set (encoder, labels) used for training
        """
        NeuralNet.train_and_eval(self, dataset, self.conv_net_model)


    def train_then_eval(self, train_loader: DataLoader, test_loader: DataLoader):
        """
            Training and evaluation of a given encoder_model, conv_vae_model, with a given set of the hyper-parameters
            :param train_loader Data loader for the training data
            :param test_loader Data loader for the evaluation data
        """
        NeuralNet.train_then_eval(self, train_loader, test_loader, self.conv_net_model)

    def __repr__(self):
        return f'Convolutional encoder_model:\n{repr(self.conv_net_model)}\nHyper-parameters:\n{repr(self.hyper_params)}'

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def feature_extractor(
            model_id: str,
            dim: int,
            in_channels: int,
            hidden_dim: int,
            out_channels: int,
            params: list) -> ConvModel:
        """
            Static method to generate a convolutional neural model with increasing number
            of input_tensor/output channels ( * 2). This convolutional model does not have connected layer
            The convolutional parameters are: kernel_size, stride, padding, batch_norm, activation.
            :param model_id: Dimension of the convolution
            :param dim: Dimension of the convolution (1 time series, 2 images, 3 video..)
            :param in_channels: Size of the latent space
            :param hidden_dim: Size of the intermediate blocks
            :param out_channels: Number of output channels
            :param params: List of convolutional parameters {kernel_size, stride, padding, batch_norm, max_pooling_kernel, activation}
        """
        assert in_channels > 0, f'z_dim {in_channels} should be > 0'
        assert hidden_dim > 1, f'hidden_dim {hidden_dim} should be > 1'
        assert out_channels > 0, f'output_dim {out_channels} should be > 0'
        assert len(params) > 1, f'Number of parameters for cascading blocks {len(params)} should be > 1'
        assert len(params[0]) == 6, f'Size of parameters {len(params[0])} should be 6'

        in_size = in_channels
        num_conv_params = len(params)
        out_size = hidden_dim
        blocks = []

        # Iteratively generate Convolution neural blocks
        for index in range(num_conv_params):
            kernel_size, stride, padding, batch_norm, max_pooling_kernel, activation = params[index]
            new_block = ConvNeuralBlock(
                dim,
                in_size,
                out_size,
                kernel_size,
                stride,
                padding,
                batch_norm,
                max_pooling_kernel,
                activation,
                False,
                False)
            blocks.append(new_block)
            in_size = out_size
            out_size = out_channels if index == num_conv_params - 2 else out_size * 2

        # Finally assemble the Convolutional model
        model = ConvModel.build(model_id, dim, blocks, None)
        del blocks
        return model
