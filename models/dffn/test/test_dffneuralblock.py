from unittest import TestCase
from models.dffn.dffneuralblock import DFFNeuralBlock
import torch


class TestDFFNeuralBlock(TestCase):
    def test_get_modules(self):
        input_size = 128
        output_size = 64
        activation = torch.nn.Tanh()
        drop_out = 0.8

        linearNeuralBlock = DFFNeuralBlock(input_size, output_size, activation, drop_out)
        print(repr(linearNeuralBlock))


