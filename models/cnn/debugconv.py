__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch

"""
    Singleton that define the various transformation for labeled input_tensor. Some labels (or classes)
    are defined as indices of input_tensor values (i.e. image)
    This class should not be instantiated.
"""


class DebugConv(object):
    def __init__(self):
        raise Exception('Should not be instantiated')

    '''
        Static method to process labels for 1D convolution. The argument, prediction is not used
        :param encoder: List of encoder of Torch input_tensor representing a encoder et
        :type encoder: List[torch.Tensor]
        :param labels: List of input_tensor labels. The labels are assumed to be class identifier
        :type labels: List[torch.Tensor]
    '''
    @staticmethod
    def text_label(features: list, labels: list, title: str) -> torch.Tensor:
        pass

    '''
        Static method to process labels for 2D convolution. 
        :param encoder: List of encoder of Torch input_tensor representing a encoder et
        :type encoder: List[torch.Tensor]
        :param labels: List of input_tensor labels. The labels are assumed to be class identifier
        :type labels: List[torch.Tensor]
        :param prediction: Predicted values (images) 
        :type prediction: torch.Tensor
    '''
    @staticmethod
    def image_label(features: list, labels: list, title: str):
        from util.imagetensor import ImageTensor
        ImageTensor.show_image(features, labels, title, 8)

