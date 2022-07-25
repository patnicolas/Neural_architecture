__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from torch import nn


class Lift(nn.Module):
    def __init__(self):
        super(Lift, self).__init__()

    def forward(self, input: torch.Tensor, sz: int) -> torch.Tensor:
        return input.view(input.size(0), -1)
