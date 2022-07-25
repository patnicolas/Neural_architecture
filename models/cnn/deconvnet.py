__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from torch.utils.data import Dataset, DataLoader
from util.plottermixin import PlotterMixin
from models.nnet.neuralnet import NeuralNet
from models.nnet.neuralnet import HyperParams
from models.cnn.deconvmodel import DeConvModel
from models.cnn.deconvneuralblock import DeConvNeuralBlock


"""
    Configurable de-convolutional neural network (convolution tranpose)
    - A de-convolutional neural network is composed of neural blocks
    - Each block is composed of Convolution transpose, Batch normalization and activation
    This class inherits from 
    - NeuralNet for training and evaluation
    - PlotterMixin for displaying 
    Note: Contrary to convolutional networks, deconvolution does not support Max pooling

    :param de_conv_net_model: de-onvolutional Neural Network encoder_model or layout as a composition of Neural blocks
    :type de_conv_net_model: cnn.deconvnetmodel.DeConvNetModel
    :param hyper_params: Training (hyper) parameters used for tuning
    :type hyper_params: nnet.hyperparams.HyperParams
    :param debug: Transform the label input_tensor prior to loss function (encoder, labels) -> converted labels
    :type debug: (list[torch.Tensor], list[torch.Tensor]) -> torch.Tensor
    :param post_epoch_func:
"""


class DeConvNet(NeuralNet, PlotterMixin):
    def __init__(self, de_conv_net_model: DeConvModel, hyper_params: HyperParams, debug):
        super(DeConvNet, self).__init__(hyper_params, debug)
        self.de_conv_net_model = de_conv_net_model

    def model_label(self) -> str:
        return self.de_conv_net_model.model_label()


    def apply_debug(self, features: list, labels: list, title: str):
        if self._debug is not None:
            self._debug(features, labels, title)

    def train_and_eval(self, dataset: Dataset):
        """
            Training and evaluation of a given encoder_model, conv_vae_model, with a given set of the hyper-parameters
            :param dataset Data set (encoder, labels) used for training
        """
        NeuralNet._train_and_eval(self, dataset, self.de_conv_net_model)


    def train_then_eval(self, train_loader: DataLoader, test_loader: DataLoader):
        """
            Training and evaluation of a given encoder_model, conv_vae_model, with a given set of the hyper-parameters
            :param train_loader Data loader for the training data
            :param test_loader Data loader for the evaluation data
        """
        NeuralNet._train_then_eval(self, train_loader, test_loader, self.de_conv_net_model)


    def __repr__(self):
        return f'DeConvolutional network:\n{repr(self.de_conv_net_model)}\nHyper-parameters:\n{repr(self.hyper_params)}'

    def __str__(self):
        return self.__repr__()


    @staticmethod
    def feature_extractor(model_id: str, conv_dimension: int, z_dim: int, hidden_dim: int, out_dim: int, params: list) -> DeConvModel:
        """
            Static method to generate a de-convolutional neural model with decreasing number
            of input_tensor/output channels (/2). The convolutional parameters are: kernel_size, stride, padding, batch_norm,
            activation.
            :param model_id: Identifier for the model used a generator
            :param conv_dimension: Dimension of the convolution (1 time series, 2 images, 3 video..)
            :param in_channels: Size of the latent space
            :param hidden_dim: Size of the intermediate blocks
            :param output_dim: Number of output channels
            :param params: List of convolutional parameters {kernel_size, stride, padding, batch_norm, activation}
        """
        assert len(params) > 1, f'Number of parameters for cascading blocks {len(params)} should be > 1'
        assert len(params[0]) == 5, f'Size of parameters {len(params[0])} should be 5'
        assert conv_dimension == 1 or conv_dimension == 2, f'Deconv conv_dim {conv_dimension} should be {1, 2}'
        assert z_dim > 0, f'Deconv z_dim {z_dim} should be > 0'
        assert hidden_dim > 1, f'Deconv hidden_dim {hidden_dim} should be > 1'
        assert out_dim > 0, f'Deconv out_dim {out_dim} should be > 0'

        in_size = z_dim
        num_conv_params = len(params)
        # Formula to compute the scaling factor to be applied to the next neural block
        block_scale_factor = 2 ** (num_conv_params - 2)
        out_size = hidden_dim * block_scale_factor
        blocks = []

        # Iteratively generate De-convolution neural blocks
        for index in range(num_conv_params):
            kernel_size, stride, padding, batch_norm, activation = params[index]
            new_block = DeConvNeuralBlock(
                conv_dimension,
                in_size,
                out_size,
                kernel_size,
                stride,
                padding,
                batch_norm,
                activation,
                False)
            blocks.append(new_block)
            # Set up the configuration for the next layer
            in_size = out_size
            # If this is the last layer then set its size as output num_tfidf_features
            # otherwise half the humber of channels
            out_size = out_dim if index == num_conv_params - 2 else out_size // 2

        model = DeConvModel(model_id, conv_dimension, blocks)
        del blocks
        return model
