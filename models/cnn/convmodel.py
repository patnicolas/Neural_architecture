__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from torch import nn
import constants
import os
from models.nnet.neuralmodel import NeuralModel
from models.cnn.convsizeparams import ConvSizeParams

"""
    Generic Convolutional neural network which can be used as Gan discriminator or VariationalNeuralBlock Auto-encoder_model
    decoder_model module. For Gan and LinearVAE, the fully connected linear modules are not defined

    :param model_id: Identifier for the model
    :param conv_dimension: Dimension of the convolution (1, 2 or 3)
    :param input_size: Size of the first convolution layer
    :param output_size: Size of output of the last convolution layer
    :param conv_model: Convolutional model (layer)
    :param dff_model_input_size: Size of input to the fully connected model
    :param dff_model: Optional fully connected model
"""


class ConvModel(NeuralModel, ConvSizeParams):
    def __init__(self,
                 model_id: str,
                 conv_dimension: int,
                 input_size: int,
                 output_size: int,
                 conv_model: nn.Sequential,
                 dff_model_input_size: int = -1,
                 dff_model: nn.Sequential = None):
        super(ConvModel, self).__init__(model_id)
        self.conv_dimension = conv_dimension
        self.input_size = input_size
        self.output_size = output_size
        self.conv_model = conv_model
        self.dff_model_input_size = dff_model_input_size
        self.dff_model = dff_model

    @classmethod
    def build(cls,
              model_id: str,
              conv_dimension: int,
              conv_neural_blocks: list,
              dff_neural_blocks: list,
              is_vae: bool = False) -> NeuralModel:
        """
            Generic Convolutional neural network which can be used as Gan discriminator or VariationalNeuralBlock Auto-encoder_model
            decoder_model module. For Gan and LinearVAE, the fully connected linear modules are not defined
            Components:
                Convolutional Neural blocks (see ConvNeuralBlock)
                Linear (Optional if input_connected_layer_size > 0)
                Activation (Optional if input_connected_layer_size > 0)
                Linear (Optional if input_connected_layer_size > 0)

            Formula to compute output_dim of a convolutional block given an in_channels
                output_dim = (in_channels + 2*padding - kernel_size)/stride + 1

            :param model_id: Identifier for the encoder_model id
            :type model_id: str
            :param conv_dimension: Dimension of the convolution (1D or 2D)
            :type conv_dimension: int
            :param conv_neural_blocks: List of convolutional neural blocks
            :type conv_neural_blocks: list
            :param dff_neural_block: Optional full connected neural block
            :type dff_neural_block: dffn.dffneuralblock.DFFNeuralBlock
            :return: Instance of type ConvModel
        """
        assert 0 < conv_dimension < 3, f'Conv net conv_dim {conv_dimension} should be {1, 2}'
        assert len(conv_neural_blocks) > 0, 'Convolutional network needs one layer'

        # Extract size of input_tensor of first convolutional layer and size of output of the last convolutional layer
        input_size = conv_neural_blocks[0].in_channels
        output_size = conv_neural_blocks[len(conv_neural_blocks) - 1]\
            .out_channels

        # Generate the convolutional modules
        conv_modules = [conv_module for conv_block in conv_neural_blocks
                        for conv_module in conv_block.modules]

        # We need to remove the last module for Variational Auto-encoder which have no activation in their
        # last layer or block of their encoder..
        if is_vae:
            conv_modules.pop()
        conv_model = nn.Sequential(*conv_modules)

        # Fully connected RBM layers for classification, It should be set to None for variational auto-encoder
        if dff_neural_blocks is not None and not is_vae:
            dff_modules = [dff_module for dff_block in dff_neural_blocks
                           for dff_module in dff_block.modules]
            dff_model_input_size = dff_neural_blocks[0].output_size
            dff_model = nn.Sequential(*tuple(dff_modules))
        else:
            dff_model_input_size = -1
            dff_model = None
        return cls(model_id, conv_dimension, input_size, output_size,
                   conv_model,dff_model_input_size, dff_model)


    def get_model(self) -> nn.Sequential:
        return self.conv_model

    def has_fully_connected(self):
        return self.dff_model is not None

    @staticmethod
    def reshape(x: torch.Tensor, resize: int) -> torch.Tensor:
        return x.view(resize, -1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
           Process the model as sequence of modules, implicitly called by __call__
           :param x: Input input_tensor
           :return: Tensor output from this network
        """
        constants.log_size(x, 'Input Conv model')
        x = self.conv_model(x)
        constants.log_size(x, 'Output Conv model')
        # If a full connected network is appended to the convolutional layers
        if self.dff_model is not None:
            constants.log_size(x, 'Before width Conv')
            sz = x.shape[0]
            x = ConvModel.reshape(x, sz)
            constants.log_size(x, 'After width Conv')
            x = self.dff_model(x)
            constants.log_size(x, 'Output connected Conv')
        return x


    def _state_params(self) -> dict:
        s = {"model_id": self.model_id, "conv_dimension": self.conv_dimension, "input_size": self.input_size,
             "output_size": self.output_size, "dff_model_input_size": self.dff_model_input_size}
        # s = "{" + f'"model_id":"{self.model_id}","conv_dimension":{self.conv_dimension},' \
        #        f'"input_size":{self.input_size},"output_size":{self.output_size},' \
        #       f'"dff_model_input_size":{self.dff_model_input_size}' + "}"
        return s


    def __repr__(self) -> str:
        modules = [module for module in self.conv_model.modules() if not isinstance(module, nn.Sequential)]
        conv_repr = ' '.join([f'\n{str(module)}' for module in modules if module is not None])
        if self.dff_model is not None:
            modules = [module for module in self.dff_model.modules() if not isinstance(module, nn.Sequential)]
            dff_repr = ' '.join([f'\n{str(module)}' for module in modules if module is not None])
            return f'{self._state_params()}{conv_repr}{dff_repr}'
        else:
            return f'{self._state_params()}{conv_repr}'


    def save(self, extra_state: dict):
        """
            Save the configuration for this convolutional neural network. The three components are
            - Configuration parameters
            - Convolutional model
            - Fully connected model
              :param extra_state Optional additional dictionary of a values
        """
        path = self.save_params(self.model_id, extra_state)
        torch.save(self.conv_model, f'{path}/{constants.conv_net_label}')
        if self.dff_model is not None:
            torch.save(self.dff_model, f'{path}/{constants.dff_net_label}')


    @staticmethod
    def load(model_id: str) -> NeuralModel:
        """
            Load this convolutional neural network from a file. he three components are
            - Configuration parameters
            - Convolutional model
            - Fully connected model
            :param model_id: Identifier for this model
            :return: Instance of this ConvModel
        """
        path, size_params = ConvSizeParams.load_params(model_id)
        conv_net = torch.load(f'{path}/{constants.conv_net_label}')
        dff_model_file = f'{path}/{constants.dff_net_label}'
        return ConvModel(model_id,
                         size_params['conv_dimension'],
                         size_params['input_size'],
                         size_params['output_size'],
                         conv_net) if not os.path.isdir(dff_model_file) else ConvModel(model_id,
                                                                                       size_params['conv_dimension'],
                                                                                       size_params['input_size'],
                                                                                       size_params['output_size'],
                                                                                       conv_net,
                                                                                       size_params['dff_model_input_size'],
                                                                                       torch.load(dff_model_file))

    def __str__(self) -> str:
        return self.__repr__()

    def model_label(self) -> str:
        return f'Convolutional neural network: {self.model_id}'


    @classmethod
    def verify(cls, neural_blocks: list):
        assert len(neural_blocks) > 0, "Convolutional neural network needs one neural block"
        for index in range(len(neural_blocks) - 1):
            assert neural_blocks[index + 1].z_dim == neural_blocks[index].output_size, \
                f'Layer {index} input_tensor != layer {index + 1} output'
