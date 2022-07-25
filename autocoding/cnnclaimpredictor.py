
import torch
from models.cnn.convnet import ConvNet
from models.cnn.convmodel import ConvModel
from models.cnn import ConvNeuralBlock
from models.nnet.hyperparams import HyperParams
from models.dffn import DFFNeuralBlock
from autocoding.tuning import Tuning
from torch.nn.modules import Tanh
from models.nnet.neuralmodel import NeuralModel
from datasets.termsctxtfidfdataset import TermsCtxTfIdfDataset


class CNNClaimPredictor(ConvNet, Tuning):
    def __init__(self,  conv_net_model: ConvModel, hyper_params: HyperParams):
        super(CNNClaimPredictor, self).__init__(conv_net_model, hyper_params, None)

    @classmethod
    def build(
            cls,
            model_id: str,
            hyper_params: HyperParams,
            dataset: TermsCtxTfIdfDataset,
            conv_output_layer_size: int,
            dff_hidden_layer_size: int) -> ConvNet:

        conv_model = CNNClaimPredictor.__create_conv_model(
            model_id,
            1,
            dataset.dimension(),
            conv_output_layer_size,
            torch.nn.ReLU(),
            CNNClaimPredictor.__create_dff_blocks(conv_output_layer_size, dataset.num_labels, [dff_hidden_layer_size])
        )
        return cls(conv_model, hyper_params)

    @staticmethod
    def __create_conv_model(
            model_id: str,
            conv_dimension: int,
            input_channels: int,
            output_channels: int,
            activation: torch.nn.Module,
            dff_blocks: list = None) -> NeuralModel:
        conv_neural_block_1 = CNNClaimPredictor.__create_conv_block(conv_dimension, input_channels, input_channels * 2, activation)
        conv_neural_block_2 = CNNClaimPredictor.__create_conv_block(conv_dimension, input_channels * 2, output_channels, activation)
        blocks = [conv_neural_block_1, conv_neural_block_2]
        return ConvModel.build(model_id, conv_dimension, blocks, dff_blocks, False)


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

    @staticmethod
    def __create_dff_blocks(dimension: int, num_labels: int, hidden_layers_size: list):
        layers_sizes = [dimension, hidden_layers_size[0], hidden_layers_size[1], num_labels] \
            if len(hidden_layers_size) > 1 \
            else \
            [dimension, hidden_layers_size[0], num_labels]

        num_layers = len(layers_sizes)
        assert num_layers > 2, f'Number of layers in this network {num_layers} should be > 2'

        # The activation of the output layer is none (pure linear)
        neural_blocks = [
            DFFNeuralBlock(layers_sizes[index], layers_sizes[index + 1], None, 0.0) if index == num_layers - 2
            else DFFNeuralBlock(layers_sizes[index], layers_sizes[index + 1], Tanh(), 0.0)
            for index in range(num_layers - 1)]
        return neural_blocks

    def compute_accuracy(self, predictionBatch: torch.Tensor, labelsBatch: torch.Tensor) -> float:
        """
            Compute the accuracy of the prediction for this particular model
            :param predictionBatch: Batch predicted features
            :param labelsBatch: Batch labels
            :return: Average accuracy for this batch
        """
        return Tuning.get_accuracy(predictionBatch, labelsBatch)