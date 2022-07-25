from unittest import TestCase

from torch import nn
import torch
import constants
import unittest
from models.nnet.hyperparams import HyperParams
from models.dffn.dffnet import DFFNet
from models.dffn.dffneuralblock import DFFNeuralBlock
from models.dffn.dffmodel import DFFModel
from datasets.numericdataset import NumericDataset


class TestDFFNet(TestCase):
    @unittest.skip("No reason")
    def test_train_and_eval(self):
        try:
            learning_rate = 0.0003
            momentum = 0.0
            epochs = 10
            optim_label = constants.optim_adam_label
            batch_size = 2
            early_stop_rate = 1.35
            loss_function = nn.CrossEntropyLoss()

            training_params = HyperParams(
                learning_rate,
                momentum,
                epochs,
                optim_label,
                batch_size,
                early_stop_rate,
                loss_function)
            input_size = 3
            output_size = 3
            activation = nn.Tanh()

            dff_layer1 = DFFNeuralBlock(input_size, output_size, activation)
            input_size_2 = 3
            output_size_2 = 2
            activation = nn.Sigmoid()
            dff_layer2 = DFFNeuralBlock(input_size_2, output_size_2, activation)

            dff_model = DFFModel([dff_layer1, dff_layer2])
            print(repr(dff_model))
            dff = DFFNet(dff_model, training_params)
            json_file = '../input_tensor/input_test2.json'
            real_dataset_2 = NumericDataset.from_json_file(json_file, '')
            dff.train_and_eval(real_dataset_2)
        except Exception as e:
            self.fail(str(e))

    def test_build_encoder(self):
        try:
            model_id = "model-1"
            input_size = 28
            hidden_dim = 9
            output_size = 2
            dff_params = [(torch.nn.ReLU(), 0.1), (torch.nn.ReLU(), 0.1), (torch.nn.Sigmoid(), -1.0)]
            dff_encoder = DFFNet.build_encoder(model_id, input_size, hidden_dim, output_size, dff_params)
            constants.get_logger().log(dff_encoder, 'info')
        except Exception as e:
            self.fail(str(e))


    def test_build_decoder(self):
        try:
            model_id = "model-1"
            input_size = 28
            hidden_dim = 9
            output_size = 2
            dff_params = [(torch.nn.ReLU(), 0.1), (torch.nn.ReLU(), 0.1), (None, -1.0)]
            dff_decoder = DFFNet.build_decoder(model_id, input_size, hidden_dim, output_size, dff_params)
            constants.get_logger().log(dff_decoder, 'info')
        except Exception as e:
            self.fail(str(e))