__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import constants
import torch

log_levels = {'ERROR': 0, 'WARN': 1, 'INFO': 2, 'PROFILE': 3, 'DEBUG': 4}

"""
    Customized logger using parameters defined in the configuration file, 'conf/config.json'. 
    The messages will be written into log file if is_print_to_log is True, on standard output otherwise
"""


class Logger(object):
    def __init__(self):
        config = constants.config
        self.log_level = log_levels[config('log_level')]
        self.is_print_to_log = config('is_print_to_log')

    def log(self, msg: str, error_level: str):
        """
            Main log method
            :param msg: Error, info or warning message
            :param error_level: Error level ('info', ...)
        """
        err_level = log_levels[error_level]
        if err_level <= self.log_level:
            self.__log_info(f'{error_level}: {msg}')

    def size(self, x: torch.Tensor, comment: str = ""):
        """
            Utility to display the shape of a input_tensor
            :param x: Torch input_tensor
            :param comment: Optional comments
        """
        if self.log_level > 2:
            assert isinstance(x, torch.Tensor), '\nNot a Tensor type'
            sz = list(x.size())
            self.__log_info(f'{str(sz)} {comment}')

    def size(self, x: torch.Tensor, y: torch.Tensor, comment: str = ""):
        """
            Utility to display the shape of a input_tensor
            :param x: Torch input_tensor
            :param comment: Optional comments
        """
        if self.log_level > 2:
            assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor), '\nNot a Tensor type'
            szx = list(x.size())
            szy = list(y.size())
            self.__log_info(f'{str(szx)} {str(szy)} {comment}')

    def __log_info(self, msg: str):
        logger.info(msg) if self.is_print_to_log else print(msg)
