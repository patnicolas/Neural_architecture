__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from models.autoencoder.vae import VAE
from models.nnet.hyperparams import HyperParams
from models.autoencoder.dffvaemodel import DFFVAEModel
import constants

"""
    Deep feed-forward variational auto-encoder have to specializes the loss function,

    :param dff_vae_model: A variational auto-encoder model implemented as PyTorch Module
    :type dff_vae_model: dffn.dffvaemodel.DFFVAEModel
    :param hyper_params: Instance of hyper-parameters for this variational auto-encoder
    :type hyper_params: nnet.hyperparams.HyperParams
    :param debug: Optional debugging method
"""


class DFFVAE(VAE):
    def __init__(self,
                 dff_vae_model: DFFVAEModel,
                 hyper_params: HyperParams,
                 debug):
        super(DFFVAE, self).__init__(dff_vae_model, hyper_params, debug)

    def loss_func(
            self,
            predicted: torch.Tensor,
            x: torch.Tensor,
            mu: torch.Tensor,
            log_var: torch.Tensor) -> float:
        criterion = self.hyper_params.loss_function
        sz = self.vae_model.encoder_model.z_dim
        try:
            # Cross-entropy for reconstruction loss for binary values
            # and MSE for continuous (TF-IDF) variable
            constants.log_size(x, 'actual before loss')
            constants.log_size(predicted, 'predict_x')
            x_value = x.view(-1, sz).squeeze(1)
            x_predict = predicted.view(-1, sz).squeeze(1)

            constants.log_size(x_value, 'x_value')
            constants.log_size(x_predict, 'x_predict')
            reconstruction_loss = criterion(x_predict, x_value)

            # Kullback-Leibler divergence for Normal distribution
            return VAE.compute_loss(reconstruction_loss, mu, log_var, sz)
        except RuntimeError as e:
            constants.log_error(f'Runtime error {str(e)}')
            return None
        except ValueError as e:
            constants.log_error(f'Value error {str(e)}')
            return None
        except KeyError as e:
            constants.log_error(f'Key error {str(e)}')
            return None