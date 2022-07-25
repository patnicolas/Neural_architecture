

from torch import nn
from architecture.architecturemodel import ArchitectureModel
from abc import abstractmethod

"""
    Generic search method for optimizing the architecture
    :param iterations: Number of iterations
"""


class SearchMethod(nn.Module):
    def __init__(self, iterations: int):
        super(SearchMethod, self).__init__()
        self.iterations = iterations

    @abstractmethod
    def forward(self, architecture_model: ArchitectureModel) -> list:
        pass
