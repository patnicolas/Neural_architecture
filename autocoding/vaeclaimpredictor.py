__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2023  All rights reserved."

import torch
from models.autoencoder.dffvae import DFFVAE
from models.autoencoder.dffvaemodel import DFFVAEModel
from models.nnet.hyperparams import HyperParams
from models.dffn import DFFModel
from models.autoencoder.variationalneuralblock import VariationalNeuralBlock
from models.dffn import DFFNeuralBlock



class VAEClaimPredictor(DFFVAE):
    def __init__(self,   dff_vae_model: DFFVAEModel, hyper_params: HyperParams):
        super(VAEClaimPredictor, self).__init__(dff_vae_model, hyper_params, False)

    @classmethod
    def build(cls,
                input_size: int,
                hidden_dim: int,
                output_size: int,
                latent_size: int,
                fc_hidden_dim: int,
                flatten_input: int,
                hyper_params: HyperParams):
        encoder_model = VAEClaimPredictor.__create_dff_encoder(input_size, hidden_dim, output_size)
        decoder_model = VAEClaimPredictor.__create_dff_decoder(output_size, hidden_dim, input_size)
        variational_block = VariationalNeuralBlock(flatten_input, fc_hidden_dim, latent_size)
        dff_vae_model = DFFVAEModel('VaeClaimPredictor', encoder_model,decoder_model,  variational_block)

        return cls(dff_vae_model, hyper_params)


    @staticmethod
    def __create_dff_encoder(input_size: int, hidden_dim: int, output_size: int) -> DFFModel:
        dff_neural_block_1 = DFFNeuralBlock(input_size, hidden_dim*4, torch.nn.ReLU(), 0.0)
        dff_neural_block_2 = DFFNeuralBlock(hidden_dim*4, hidden_dim*2, torch.nn.ReLU(), 0.0)
        dff_neural_block_3 = DFFNeuralBlock(hidden_dim *2, output_size, torch.nn.ReLU(), 0.0)
        return DFFModel("DFF encoder", [dff_neural_block_1, dff_neural_block_2, dff_neural_block_3])

    @staticmethod
    def __create_dff_decoder(input_size: int, hidden_dim: int, output_size: int) -> DFFModel:
        dff_neural_block_1 = DFFNeuralBlock(input_size, hidden_dim*2, torch.nn.ReLU(), 0.0)
        dff_neural_block_2 = DFFNeuralBlock(hidden_dim *2, hidden_dim * 4, torch.nn.ReLU(), 0.0)
        dff_neural_block_3 = DFFNeuralBlock(hidden_dim *4, output_size, torch.nn.ReLU(), 0.0)
        return DFFModel("DFF decoder", [dff_neural_block_1, dff_neural_block_2, dff_neural_block_3])