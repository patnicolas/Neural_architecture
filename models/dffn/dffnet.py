__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from models.dffn.dffmodel import DFFModel
from models.dffn.dffneuralblock import DFFNeuralBlock
from models.nnet.neuralnet import NeuralNet
from models.nnet.hyperparams import HyperParams
from torch.utils.data import Dataset, DataLoader


"""
    Implementation of Deep feed-forward (full-connected) neural network as a sub-class of nnet.neuralnet.NeuralNet
    There are 3 constructors
    __init__:  Generic constructor
    build_encoder: Create a pre-canned feed-forward model for encoding of variational auto-encoder or generative
                adversarial network discriminator
    build_decoder: Create a pre-canned feed-forward model for decoder of variational auto-encoder or generative
                adversarial network generator
    
    :param dff_model: Model associated with the feed-forward network
    :type dff_model: dffn.dffmodel.DFFModel
    :param hyper_params: Model hyper-parameters used in training
    :type hyper_params: nnet.hyperparams.HyperParams
    :param debug: Optional debugging function
"""


class DFFNet(NeuralNet):
    def __init__(self, dff_model: DFFModel, hyper_params: HyperParams, debug):
        super(DFFNet, self).__init__(hyper_params, debug)
        self.dff_model = dff_model


    def apply_debug(self, features, labels, prediction: torch.Tensor) -> torch.Tensor:
        pass


    def model_label(self) -> str:
        return self.dff_model.model_id


    def train_and_eval(self, dataset: Dataset):
        """
            Polymorphic method to train and evaluate a NN model
            :param dataset:  Training and evaluation data set
            :return: None
        """
        self._train_and_eval(dataset, self.dff_model)


    def train_then_eval(self, train_loader: DataLoader, test_loader: DataLoader):
        """
            Polymorphic method to train then evaluate a NN model
            :param train_loader: Loader for the training set
            :param test_loader: Loader for the evaluation or test1 set
            :return: None
        """
        self._train_then_eval(train_loader, test_loader, self.dff_model)


    def __repr__(self) -> str:
        return repr(self.dff_model) + '\n' + repr(self.hyper_params)


    @staticmethod
    def build_encoder(
            model_id: str,
            input_size: int,
            hidden_dim: int,
            output_size: int,
            dff_params: list) -> DFFModel:
        """
            Build a feed-forward network model as VAE decoder or Gan generator
            :param model_id: Identifier for the decoder
            :type model_id: str
            :param input_size: Size of the connected input_tensor layer
            :type input_size: int
            :param hidden_dim: Size of the last hidden layer. The size of previous layers are halved from the
                    previous layer
            :type hidden_dim: int
            :param output_size: Size of the output layer
            :type output_size: int
            :param dff_params: List of parameters tuple{ (activation_func, drop-out rate)
            :type dff_params: list
        """
        DFFNet.__validate("DFF encoder", input_size, hidden_dim, output_size, dff_params)
        in_size = input_size
        num_dff_params = len(dff_params)
        # Compute the size of the input_tensor layer given the size lf the last hidden layer
        scale_factor = 2 ** (num_dff_params  - 2)
        out_size = hidden_dim*scale_factor
        blocks = []

        # Iteratively generate De-convolution neural blocks
        for index in range(num_dff_params):
            activation, drop_out = dff_params[index]
            # build the neural block from the parameter tuple
            dff_neural_block = DFFNeuralBlock(in_size, out_size, activation, drop_out)
            blocks.append(dff_neural_block)
            if index == num_dff_params - 2:
                out_size = output_size
            else:
                out_size = out_size // 2
        dff_model = DFFModel(model_id, blocks)
        del blocks
        return dff_model


    @staticmethod
    def build_decoder(
            model_id: str,
            input_size: int,
            hidden_dim: int,
            output_size: int,
            dff_params: list) -> DFFModel:
        """
            Build a feed-forward network model as VAE decoder or Gan generator
            :param model_id: Identifier for the decoder
            :type model_id: str
            :param input_size: Size of the connected input_tensor layer
            :type input_size: int
            :param hidden_dim: Size of the first hidden layers. The size of subsequent layers are doubled from the
                       previous layer
            :type hidden_dim: int
            :param output_size: Size of the output layer
            :type output_size: int
            :param dff_params: List of parameters tuple{ (activation_func, drop-out rate)
            :type dff_params: list
        """
        DFFNet.__validate("DFF decoder", input_size, hidden_dim, output_size, dff_params)
        in_size = input_size
        num_dff_params = len(dff_params)
        out_size = hidden_dim
        blocks = []

        # Iteratively generate De-convolution neural blocks
        for index in range(num_dff_params):
            activation, drop_out = dff_params[index]
            dff_neural_block = DFFNeuralBlock(in_size, out_size, activation, drop_out)
            blocks.append(dff_neural_block)
            if index == num_dff_params - 2:
                out_size = output_size
            else:
                out_size = out_size * 2
        dff_model = DFFModel(model_id, blocks)
        del blocks
        return dff_model

    @staticmethod
    def __validate(
            method_label: str,
            input_size: int,
            hidden_dim: int,
            output_size: int,
            dff_params: list):
        assert len(dff_params) > 1, f'{method_label} Number of parameters for cascading blocks {len(dff_params)} should be > 1'
        assert len(dff_params[0]) == 2, f'{method_label} Size of parameters {len(dff_params[0])} should be 2'
        assert input_size > 0, f'{method_label} input_dim {input_size} should be > 0'
        assert hidden_dim > 1, f'{method_label} hidden_dim {hidden_dim} should be > 1'
        assert output_size > 0, f'{method_label} output_dim {output_size} should be > 0'
