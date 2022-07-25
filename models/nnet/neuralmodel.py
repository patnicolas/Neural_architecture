__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from abc import abstractmethod

"""
    Abstract base class for Neural network models. The sub-classes have to implement get_model
    method.
    :param model_id: Identifier for this model
    :type model_id: str
"""


class NeuralModel(torch.nn.Module):
    def __init__(self,  model_id: str):
        super(NeuralModel, self).__init__()
        self.model_id = model_id

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __repr__(self) -> str:
        return f'Model: {self.model_id}'

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def save(self, extra_params: dict = None):
        pass

