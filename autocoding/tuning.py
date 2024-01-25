__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2023  All rights reserved."

import torch


class Tuning(object):

    @staticmethod
    def get_accuracy(predictionBatch: torch.Tensor, labelsBatch: torch.Tensor) -> (float, list):
        """
            Compute the accuracy of the prediction for this particular model
            :param predictionBatch: Batch predicted features
            :param labelsBatch: Batch labels
            :return: Average accuracy for this batch
        """
        accuracy = 0.0
        buf = []
        for predicted, label in zip(predictionBatch, labelsBatch):
            index = torch.argmax(predicted)
            if index == label:
                accuracy += 1.0
            buf.append(f'{index},{label}')
        return accuracy / len(labelsBatch), buf