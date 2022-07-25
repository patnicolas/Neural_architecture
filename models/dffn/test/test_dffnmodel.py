
import torch
from unittest import TestCase
from models.dffn.dffneuralblock import DFFNeuralBlock
from models.dffn.dffmodel import DFFModel

class TestDFFNModel(TestCase):
    def test_forward(self):
        dff_neural_block1 = TestDFFNModel.__createNeuralBlock(torch.nn.Tanh())
        dff_neural_block2 = TestDFFNModel.__createNeuralBlock(torch.nn.Sigmoid())
        dff = DFFModel([dff_neural_block1, dff_neural_block2])
        print(repr(dff))
        t = torch.tensor([[0.8000, 0.7000, 0.9000],
                          [0.2000, 0.0000, 0.1000],
                          [0.6000, 1.0000, 0.8000]])
        t2 = dff.forward(t)
        print(t2.detach())


    @staticmethod
    def __createNeuralBlock(activation: torch.nn.Module) -> DFFNeuralBlock:
        input_size = 3
        output_size = 3
        return DFFNeuralBlock(input_size, output_size, activation, 0.0)
