__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from models.nnet.neuralmodel import NeuralModel
from models.cnn.convsizeparams import ConvSizeParams
from models.cnn.deconvneuralblock import DeConvNeuralBlock
import constants
import torch

"""
    Implementation of a de-convolutional neural model.
    Contrary to convolutional neural networks, convolutional networks are used as part 
    of Gan generator or LinearVAE decoder_model and therefore do not contain fully or linear layers.
    There are 3 constructors:
        - Default __init__ using PyTorch module
        - Build using pre-defined de-convolutional neural blocks
        - mirror using an existing convolutional neural blocks (used in VAE and GANs)

    :param model_id: Identifier for the encoder_model id
    :param conv_dimension: Dimension of the convolution (1 to 3)
    :param input_size: Size of the input to the first de-convolutional layer
    :param output_size: Size of the output of the last de-convolutional layer
    :param de_conv_model: De-deconvolutional model using PyTorch modules
"""


class DeConvModel(NeuralModel, ConvSizeParams):
    def __init__(self,
                 model_id: str,
                 conv_dimension: int,
                 input_size: int,
                 output_size: int,
                 de_conv_model: torch.nn.Sequential):
        super(DeConvModel, self).__init__(model_id)
        self.conv_dimension = conv_dimension
        self.input_size = input_size
        self.output_size = output_size
        self.de_conv_model = de_conv_model


    @classmethod
    def build(cls, model_id: str, de_conv_neural_blocks: list):
        """
            The de-convolutional neural network is a mirror implementation of the convolutional neural network
            Important note: Contrary to convolutional neural networks, convolutional networks are used as part
            of Gan generator or LinearVAE decoder_model and therefore do not contain fully or linear layers.
            There are 2 constructors:
            - Default __init__ using predefined de-convolutional neural blocks
            - mirror using existing convolutional neural blocks (used in VAE and GANs)

            Formula for processing neural block with in_channels
                output_dim = stride*(in_channels -1) - 2*padding + kernel_size

            :param model_id: Identifier for the encoder_model id
            :param de_conv_neural_blocks: List of convolutional neural blocks
        """
        assert len(de_conv_neural_blocks) > 0, 'De-convolutional model needs one layer'

        conv_dimension = de_conv_neural_blocks[0].conv_dimension
        input_size = de_conv_neural_blocks[0].in_channels
        output_size = de_conv_neural_blocks[len(de_conv_neural_blocks) - 1].out_channels
        conv_modules = tuple([conv_module for conv_block in de_conv_neural_blocks
                              for conv_module in conv_block.modules
                              if conv_module is not None])
        de_conv_model = torch.nn.Sequential(*conv_modules)
        return cls(model_id, conv_dimension, input_size, output_size, de_conv_model)


    @classmethod
    def build_for_vae(cls,
                      model_id: str,
                      conv_neural_blocks: list,
                      last_activation: torch.nn.Module = None) -> NeuralModel:
        """
            Build a mirror De-convolutional neural model from existing set of convolutional neural blocks for which
            the number of input_tensor channels and the output activation function have to be overridden.
            :param model_id: Identifier for the model
            :param conv_neural_blocks: Convolutional neural blocks used to build the Convolutional model (ConvModel)
            :param disc_in_channels: Number of input_tensor channels to the De-convolutional model
            :param last_activation: Activation function for the last layer of the De-convolutional model
            :return: De-convolutional neural model
        """
        de_conv_neural_blocks = []

        # Need to reverse the order of convolutional neural blocks
        last_out_channels = conv_neural_blocks[0].in_channels
        list.reverse(conv_neural_blocks)
        in_channels = conv_neural_blocks[0].out_channels // 2

        for idx in range(len(conv_neural_blocks)):
            conv_neural_block = conv_neural_blocks[idx]
            # If this is the first block
            if idx == 0:
                de_conv_neural_block = DeConvNeuralBlock.transpose_vae(
                    conv_neural_block,
                    in_channels,
                    conv_neural_block.out_channels)
                in_channels = conv_neural_block.out_channels
            # If this is the last block
            elif idx == len(conv_neural_blocks) - 1:
                de_conv_neural_block = DeConvNeuralBlock.transpose_vae(
                    conv_neural_block,
                    in_channels,
                    last_out_channels,
                    last_activation)
            else:
                de_conv_neural_block = DeConvNeuralBlock.transpose_vae(conv_neural_block, in_channels)
                in_channels = in_channels // 2
            de_conv_neural_blocks.append(de_conv_neural_block)

        de_conv_model = DeConvModel.build(model_id, de_conv_neural_blocks)
        del de_conv_neural_blocks
        return de_conv_model


    @classmethod
    def build_for_gan(cls,
                      model_id: str,
                      conv_neural_blocks: list,
                      in_channels: int,
                      out_channels: int = None,
                      last_block_activation: torch.nn.Module = None) -> NeuralModel:
        """
            Build a mirror De-convolutional neural model from existing set of convolutional neural blocks for which
            the number of input_tensor channels and the output activation function have to be overridden.
            :param model_id: Identifier for the model
            :param conv_neural_blocks: Convolutional neural blocks used to build the Convolutional model (ConvModel)
            :param in_channels: Number of input_tensor channels to the De-convolutional model
            :param out_channels: Number of input_tensor channels to the De-convolutional model
            :param last_block_activation: Activation function for the last layer of the De-convolutional model
            :return: De-convolutional neural model
        """
        de_conv_neural_blocks = []

        # Need to reverse the order of convolutional neural blocks
        list.reverse(conv_neural_blocks)

        for idx in range(len(conv_neural_blocks)):
            conv_neural_block = conv_neural_blocks[idx]
            new_in_channels = None
            activation = None
            last_out_channels = None

            # Will update number of input_tensor channels for the first de-convolutional layer
            if idx == 0:
                new_in_channels = in_channels
            # Defined, if necessary the activation function for the last layer of the de-convolutional layer
            elif idx == len(conv_neural_blocks) - 1:
                if last_block_activation is not None:
                    activation = last_block_activation
                if out_channels is not None:
                    last_out_channels = out_channels

            # Apply transposition to the convolutional block to generate the mirrored de-convolution block
            de_conv_neural_block = DeConvNeuralBlock.transpose_gan(conv_neural_block,
                                                                   new_in_channels,
                                                                   last_out_channels,
                                                                   activation)
            de_conv_neural_blocks.append(de_conv_neural_block)
        de_conv_model = DeConvModel.build(model_id, de_conv_neural_blocks)
        del de_conv_neural_blocks
        return de_conv_model


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Process input_tensor data through the model as sequence of modules, implicitly called by __call__
            :param x: Input input_tensor
            :return: Tensor output from this network
        """
        constants.log_size(x, 'Input DeConv model')
        x = self.de_conv_model(x)
        constants.log_size(x, 'Output DeConv model')
        return x

    def get_model(self) -> torch.nn.Sequential:
        return self.de_conv_model

    def _state_params(self) -> dict:
        return {"model_id":self.model_id,"conv_dimension":self.conv_dimension,"input_size":self.input_size,"output_size":self.output_size}

    def __repr__(self) -> str:
        modules = [module for module in self.de_conv_model.modules() if not isinstance(module, torch.nn.Sequential)]
        return ' '.join([f'\n{str(module)}' for module in modules if module is not None])

    def save(self, extra_state: dict):
        """
            Save the configuration for this de-convolutional neural network. The 2 components are
            - Configuration parameters
            - De-convolutional model
            :param extra_state Optional additional dictionary of a values
        """
        path = self.save_params(self.model_id, extra_state)
        torch.save(self.de_conv_model, f'{path}/{constants.de_conv_net_label}')

    @staticmethod
    def load(model_id: str) -> NeuralModel:
        """
            Load this de-convolutional neural network from a file. The two components are
            - Configuration parameters
            - De-Convolutional model
            :param model_id: Identifier for this model
            :return: Instance of this ConvModel
        """
        path, size_params = ConvSizeParams.load_params(model_id)
        de_conv_net = torch.load(f'{path}/{constants.de_conv_net_label}')
        return DeConvModel(model_id,
                           size_params['conv_dimension'],
                           size_params['input_size'],
                           size_params['output_size'],
                           de_conv_net)
