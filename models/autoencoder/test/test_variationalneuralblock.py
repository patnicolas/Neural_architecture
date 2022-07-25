from unittest import TestCase

import torch
import constants
from models.autoencoder.variationalneuralblock import VariationalNeuralBlock
from datasets.numericdataset import NumericDataset
from torch.utils.data import DataLoader


class TestVariationalNeuralBlock(TestCase):
    def test___init(self):
        input_size = 32
        hidden_dim = 16
        latent_size = 12
        variational_block = VariationalNeuralBlock(input_size, hidden_dim, latent_size)
        print(repr(variational_block))

    def test_forward(self):
        input_size = 32
        hidden_dim = 16
        latent_size = 12
        variational_block = VariationalNeuralBlock(input_size, hidden_dim, latent_size)
        try:
            x = torch.ones(input_size)
            x[2] = 0.5
            x[5] = 0.5
            x[9] = 0.5
            print(str(x))
            y = variational_block(x)
            print(str(y.detach()))
        except Exception as e:
            self.fail(str(e))

    def test_forward_loader(self):
        input_size = 32
        hidden_dim = 16
        latent_size = 12
        variational_block = VariationalNeuralBlock(input_size, hidden_dim, latent_size)
        try:
            batch_size = 4
            features = torch.ones(input_size*batch_size)
            target = torch.ones(batch_size)
            x = torch.cat((features, target), dim=0)
            constants.log_size(x, 'checkpoint')
            x[2] = 0.5
            x[5] = 0.5
            x[9] = 0.5
            t = x.view((batch_size, -1))
            constants.log_size(t, 't')
            ds = NumericDataset.from_tensor(t)
            data_loader = DataLoader(ds, batch_size)
            for idx, (features, target) in enumerate(data_loader):
                print(str(features))
                y = variational_block(features)
                print(str(y.detach()))
        except Exception as e:
            self.fail(str(e))





