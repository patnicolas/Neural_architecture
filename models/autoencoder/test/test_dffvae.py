from unittest import TestCase

import torch
from models.nnet.hyperparams import HyperParams
from models.autoencoder.variationalneuralblock import VariationalNeuralBlock
from models.dffn.dffneuralblock import DFFNeuralBlock
from models.dffn.dffmodel import DFFModel
from models.autoencoder.dffvaemodel import DFFVAEModel
from torch.utils.data import DataLoader
from models.autoencoder.dffvae import DFFVAE
from datasets.numericdataset import NumericDataset
from models.nnet.neuralnet import NeuralNet


class TestDFFVAE(TestCase):
    def test_train_and_eval(self):
        try:
            input_size = 49
            hidden_dim = 8
            output_size = 20
            dff_encoder = TestDFFVAE.__create_dff_encoder(input_size, hidden_dim, output_size)
            dff_decoder = TestDFFVAE.__create_dff_decoder(output_size, hidden_dim, input_size)
            latent_size = 16
            fc_hidden_dim = 24
            flatten_input = 20
            variational_block = VariationalNeuralBlock(flatten_input, fc_hidden_dim, latent_size)

            dff_vae_model = DFFVAEModel("dff_vae_model", dff_encoder, dff_decoder, variational_block)
            lr = 0.001
            momentum = 0.9
            epochs = 20
            optim_label = 'adam'
            batch_size = 14
            early_stop_patience = 3
            loss_func = torch.nn.BCELoss(reduction='sum')
            hyper_params = HyperParams(lr, momentum, epochs, optim_label, batch_size, early_stop_patience, loss_func)

            # Step 3: Load data set
            train_loader, test_loader = TestDFFVAE.__load_data(batch_size)

            dff_vae = DFFVAE(dff_vae_model, hyper_params, None)
            print(repr(dff_vae))
            dff_vae._train_and_eval(train_loader, test_loader)
        except Exception as e:
            self.fail(str(e))


            # -----------------  Supporting methods ------------------

    @staticmethod
    def __create_dff_encoder(input_size: int, hidden_dim: int, output_size: int) -> DFFModel:
        dff_neural_block_1 = DFFNeuralBlock(input_size, hidden_dim*4, torch.nn.ReLU(), 0.2)
        dff_neural_block_2 = DFFNeuralBlock(hidden_dim*4, hidden_dim*2, torch.nn.ReLU(), 0.2)
        dff_neural_block_3 = DFFNeuralBlock(hidden_dim *2, output_size, torch.nn.ReLU(), 0.2)
        return DFFModel("DFF encoder", [dff_neural_block_1, dff_neural_block_2, dff_neural_block_3])

    @staticmethod
    def __create_dff_decoder(input_size: int, hidden_dim: int, output_size: int) -> DFFModel:
        dff_neural_block_1 = DFFNeuralBlock(input_size, hidden_dim*2, torch.nn.ReLU(), 0.2)
        dff_neural_block_2 = DFFNeuralBlock(hidden_dim *2, hidden_dim * 4, torch.nn.ReLU(), 0.2)
        dff_neural_block_3 = DFFNeuralBlock(hidden_dim *4, output_size, torch.nn.ReLU(), 0.2)
        return DFFModel("DFF decoder", [dff_neural_block_1, dff_neural_block_2, dff_neural_block_3])

    @staticmethod
    def __load_data(batch_size: int) -> (DataLoader, DataLoader):
        import constants

        x = torch.arange(0.0, 50.0, 0.001, dtype=torch.float32).view(-1, 50)
        constants.log_info(f'num data points {len(x)}')
        constants.log_size(x, 'Input actual')
        numeric_dataset = NumericDataset.from_tensor(x)
        constants.log_size(x, 'Load data for DFF VAE')
        return NeuralNet.init_data_loader(batch_size, numeric_dataset)
