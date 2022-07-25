__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from torch import nn
from models.nnet.neuralmodel import NeuralModel

"""
    Generic Deep Feed Forward Network encoder_model 
    :param model_id: Identifier for this DFF encoder_model
    :type model_id: str 
    :param dff_neural_blocks: List of modules 
    :type dff_neural_blocks: lst
"""


class DFFModel(NeuralModel):
    def __init__(self, model_id: str, dff_neural_blocks: list):
        super(DFFModel, self).__init__( model_id)
        assert len(dff_neural_blocks), 'DFF model should have at least 1 neural block'

        # Define the sequence of modules from the layout
        self.input_size = dff_neural_blocks[0].input_size
        self.output_size = dff_neural_blocks[len(dff_neural_blocks) - 1].output_size
        modules = [module for layer in dff_neural_blocks for module in layer.modules]
        self.model = nn.Sequential(*modules)

    def set_model_id(self, new_model_id: str):
        """
            Update dynamically the model identifier
            :param new_model_id: New model identifier
            :return: NNone
        """
        self.model_id = new_model_id

    def get_model(self) -> nn.Sequential:
        return self.model

    def label(self) -> str:
        return f'Deep Feed Forward Network {self.model_id}'

    def __repr__(self) -> str:
        modules = [module for module in self.model.modules() if not isinstance(module, nn.Sequential)]
        net_repr = ' '.join([f'\n   {str(module)}' for module in modules])
        return f'\n    --- Input layer size: {self.input_size}  Output layer size: {self.output_size}{net_repr}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Process input_tensor data through the model as sequence of modules, implicitly called by __call__
            :param x: Input input_tensor
            :return: Tensor output from this network
        """
        return self.model(x)

    @staticmethod
    def __verify(neural_blocks: list):
        assert len(neural_blocks) > 0, "Deep Feed Forward network needs at least one layer"
        for index in range(len(neural_blocks) - 1):
            assert neural_blocks[index + 1].z_dim == neural_blocks[index].output_size, \
                f'Layer {index} input_tensor != layer {index+1} output'
