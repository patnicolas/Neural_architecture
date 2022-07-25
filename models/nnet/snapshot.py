__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from util.s3util import S3Util

"""
    Implementation of a snapshot for training PyTorch encoder_model. The snapshot consists of three components
    - Model parameters (ordered dictionary)
    - Optimizer parameters (ordered dictionary)
    - Current epoch
    :param s3_util  Instance of s3 utility to read, write from and to AWS S3 bucket
"""


class Snapshot(object):
    def __init__(self, s3_util: S3Util):
        self.s3_util = s3_util

    def put(self, model: torch.nn.Module, optimizer: torch.nn.Module, epoch: int):
        self.s3_util.write_ordered_dict(model.state_dict(), 'mod')
        self.s3_util.write_ordered_dict(optimizer.state_dict(), 'opt')
        self.s3_util.write_value(str(epoch), 'cnt')

    def get(self, model: torch.nn.Module, optimizer: torch.nn.Module) -> (torch.nn.Module, torch.nn.Module, int):
        model_ordered_dict = self.s3_util.read_ordered_dict('mod')
        optimizer_ordered_dict = self.s3_util.read_ordered_dict('opt')
        epoch = self.s3_util.read_value('cnt')
        model.load_state_dict(model_ordered_dict)
        optimizer.load_state_dict(optimizer_ordered_dict)
        return model, optimizer, int(epoch)
