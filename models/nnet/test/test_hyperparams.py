from unittest import TestCase

import torch
from models.nnet.hyperparams import HyperParams
from models.dffn import DFFNeuralBlock


class TestHyperParams(TestCase):
    def test_test_conversion(self):
        try:
            def f(x: torch.Tensor) -> torch.Tensor:
                return x+0.1
            y = HyperParams.test_conversion(f)
            print(y)
        except Exception as e:
            self.fail(str(e))

    def test_initialize_weight(self):
        lr = 0.00
        momentum = 0.9
        epochs = 20
        optim_label= 'adam'
        batch_size = 16
        early_stop_patience = 3
        loss_function = torch.nn.MSELoss()
        normal_weight_initialization = True
        dff_modules = TestHyperParams.__get_modules()

        hyper_params = HyperParams(
            lr,
            momentum,
            epochs,
            optim_label,
            batch_size,
            early_stop_patience,
            loss_function,
            normal_weight_initialization)

        print([str(m.weight) for m in dff_modules if type(m) == torch.nn.Linear])
        hyper_params.initialize_weight(dff_modules)
        print([str(m.weight) for m in dff_modules if type(m) == torch.nn.Linear])

    @staticmethod
    def __get_modules():
        input_size = 128
        output_size = 64
        activation = torch.nn.Tanh()
        drop_out = 0.8
        linear_neural_block = DFFNeuralBlock(input_size, output_size, activation, drop_out)
        return linear_neural_block.get_modules()
