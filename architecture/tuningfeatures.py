__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2023  All rights reserved."
import numpy as np
import math


"""
    Wraps a tuning param. An AssertError is thrown if the lower and upper bounds are incorrectly defined
    
    :param param_name: Name of the parameter
    :param param_type:  Type of the parameter {real, ordinal}
    :param to_normalize: Is this parameter needs to be normalized
    :param lower_bound: Minimum value this parameter may have
    :param upper_bound: Maximum value this parameter may have
"""

class TuningParam(object):
    def __init__(self, param_name: str, param_type: str, to_normalize: bool, lower_bound: float, upper_bound: float):
        assert upper_bound > lower_bound, f'Min {lower_bound} should be < Max {upper_bound}'
        self.param_name = param_name
        self.param_type = param_type
        self.to_normalize = to_normalize
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.param_value = 0.5*(upper_bound + lower_bound)

    def __call__(self) -> float:
        return (self.param_value - self.lower_bound) / (self.upper_bound - self.lower_bound) \
            if self.to_normalize \
            else \
            self.param_value

    def update(self, new_value: float):
        """
            Update the value of this tuning parameter. The update should take into account whether the value of
            this parameters is to be normalized and if it exceeds its bounds
            :param new_value: New value computed from the optimizer
            :return: None
        """
        actual = new_value*(self.upper_bound - self.lower_bound) + self.lower_bound if self.to_normalize else new_value
        if actual > self.upper_bound:
            actual = self.upper_bound
        elif actual < self.lower_bound:
            actual = self.lower_bound

        if self.param_type == 'ordinal':
            floor_value = math.floor(actual)
            ceil_value = math.ceil(actual)
            self.param_value = floor_value if actual - floor_value < ceil_value - actual else ceil_value
        else:
            self.param_value = actual

    def __str__(self) -> str:
        return f'Param: {self.param_name}, Type: {self.param_type}, Value: {self.param_value}, ' \
               f'Lower bound: {self.lower_bound}, Upper bound: {self.upper_bound}'


"""
    Manage a collection of tuning features to generate a numpy array of normalized features
"""


class TuningFeatures(object):

    def __init__(self):
        self.parameters = {}

    def add_param(self, param: TuningParam):
        self.parameters[param.param_name] = param

    def __len__(self) -> int:
        return len(self.parameters)

    def __str__(self) -> str:
        return str(self.parameters)

    def get_param(self, key: str) -> TuningParam:
        return self.parameters[key]

    def get_params(self) -> list:
        return list(self.parameters.values())

    def get_values(self) -> list:
        return [param() for param in self.parameters.values()]

    def sort(self):
        self.parameters = dict(sorted(self.parameters.items()))

    def update(self, weights: np.array) -> np.array:
        for value, weight in zip(self.parameters.values(), weights):
            value.update(weight)
        np.array(TuningFeatures.__execute())

    def __str__(self) -> str:
        param_str = [str(param) for param in self.parameters.values()]
        return '\n'.join(param_str)

    @staticmethod
    def __execute() -> float:
        return np.random.random_sample(1)
