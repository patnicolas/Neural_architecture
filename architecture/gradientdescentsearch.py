

import torch
from architecture.architecturemodel import ArchitectureModel
from architecture.searchmethod import SearchMethod
from torch import nn
from torch.autograd import Variable

"""
    Search using Gradient descent
    :param learning_rate: learning rate
    :param momentum for this gradient descent
    :param iterations: Number of iterations
"""


class GradientDescentSearch(SearchMethod):
    def __init__(self, learning_rate: float, momentum: float, iterations: int):
        super(GradientDescentSearch, self).__init__(iterations)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def forward(self, architecture_model: ArchitectureModel) -> list:
        """
            Execution of the model give the current architecture parameters
            :param architecture_model: Model for optimizing the architecture
            :return: History of losses
        """
        losses = []

        # Get the initial weights
        weights = nn.Parameter(architecture_model.initialize())
        for iteration in range(self.iterations):
            # Call ArchitectureModel.forward
            error_tensor = architecture_model(weights)
            # Compute the derivatives
            error_tensor.backward()

            # Initialize the rate of change for momentum
            with torch.no_grad():
                # Apply the gradient descent formula
                weights = self.__step(weights)
                if weights is not None:
                    for idx in range(len(weights)):
                        weights.grad[0] = 0.0

            losses.append(error_tensor.item())

    def __step(self, weights: torch.Tensor) -> torch.Tensor:
        """
            Apply the Gradient formula with momentum if
            :param weights: Weight used in the configuration of the architecture
            :return: Updated weights
        """
        if self.momentum > 0.0:
            rate_of_change = torch.zeros(len(weights))
        for idx in range(len(weights)):
            momentum = self.momentum * rate_of_change[idx] if self.momentum > 0.0 else 0.0
            delta = self.learning_rate * weights.grad[idx] + momentum
            weights[idx] -= delta
            rate_of_change[idx] = delta
        return weights

    def __str__(self):
        return f'Gradient descent with learning rate {self.learning_rate}, momentum {self.momentum} ' \
               f'and {self.iterations} iterations'

    @staticmethod
    def __loss_func(score_tensor: torch.Tensor) -> torch.Tensor:
        return 1.0 - score_tensor





