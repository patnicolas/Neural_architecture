__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2023  All rights reserved."

from architecture.architecturemodel import ArchitectureModel
from architecture.searchmethod import SearchMethod

"""
    Controller for optimizing the configuration of Kafka and Spark for low latency
    :param search_method: Search or optimization method
    :param architecture_filename: Name of the S3 file containing the architecture parameters
"""


class Controller(object):
    def __init__(self, search_method: SearchMethod, architecture_filename: str):
        tuning_features = ArchitectureModel.load(architecture_filename)
        self.architecture_model = ArchitectureModel(architecture_filename, 1.0, tuning_features)
        self.search_method = search_method

    def search(self) -> list:
        """
            Execute the optimization
            :return: History of losses collected during optimization
        """
        return self.search_method(self.architecture_model)

    def __str__(self) -> str:
        return f'\nSearch method {str(self.search_method)}\nModel: {str(self.architecture_model)}'



