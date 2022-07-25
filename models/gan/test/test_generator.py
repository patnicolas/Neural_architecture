from unittest import TestCase
import torch
from models.gan.generator import Generator
import constants


class TestGenerator(TestCase):
    def test_build_from_conv(self):
        try:
            dim = 2
            z_dim = 10
            hidden_dim = 13
            out_dim = 1
            params = [(3, 2, 1, True, torch.nn.ReLU()), (2, 2, 0, True, torch.nn.ReLU()),
                      (2, 2, 0, False, torch.nn.Tanh())]
            gen = Generator.build_from_de_conv('model-1', dim, z_dim, hidden_dim, out_dim, params)
            constants.log_info(repr(gen))
        except Exception as e:
            self.fail(str(e))

    def test_build_from_dff(self):
        try:
            model_id = "model-1"
            input_size = 28
            hidden_dim = 9
            output_size = 2
            dff_params = [(torch.nn.ReLU(), 0.1), (torch.nn.ReLU(), 0.1), (torch.nn.Sigmoid(), -1.0)]
            gen = Generator.build_from_dff(model_id, input_size, hidden_dim, output_size, dff_params)
            constants.log_info(repr(gen))
        except Exception as e:
            self.fail(str(e))


    def test_noise(self):
        try:
            num_samples = 120
            z_dim = 18
            rand_tensor = Generator.noise(num_samples, z_dim)
            constants.log_info(rand_tensor)
        except Exception as e:
            self.fail(str(e))

