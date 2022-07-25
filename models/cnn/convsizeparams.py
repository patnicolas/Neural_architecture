__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import abc
import constants
from util.ioutil import IOUtil
import os

"""
    Encapsulate the sizing parameters for this convolutional or de-convolutional model
"""


class ConvSizeParams(object):
    @abc.abstractmethod
    def _state_params(self):
        pass

    def save_params(self, model_id: str, extra_state: dict) -> str:
        """
            Save the configuration size parameters for this convolutional or de-convolutional model
            :param extra_state: Optional additional dictionary of a values
            :return relative path the data is saved
         """
        import json

        state_params = {**self._state_params(), **extra_state} if extra_state is not None else self._state_params()

        # Create a path if it does not exist
        path = f'{constants.models_path}/{model_id}'
        if not os.path.isdir(path):
            os.mkdir(path)
        # store the parameters into file
        ioutil = IOUtil(f'{path}/{constants.params_label}')
        ioutil.from_text(json.dumps(state_params))
        return path

    @staticmethod
    def load_params(model_id: str) -> (str, dict):
        """
            Load the sizing parameter for this neural model
            :param model_id: Identifier for the neural model
            :return: Pair (relative path the params are loaded from, dictionary of parameters)
        """
        path = f'{constants.models_path}/{model_id}'
        ioutil = IOUtil(f'{path}/{constants.params_label}')
        return path, ioutil.to_json()