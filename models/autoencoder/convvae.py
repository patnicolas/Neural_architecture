__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from models.autoencoder.convvaemodel import ConvVAEModel
from models.nnet.hyperparams import HyperParams
import constants
from models.autoencoder.vae import VAE

"""
    Convolutional variational auto-encoder have to specializes the loss function unique to convolutional networkds
    Design patterns: Bridge

    :param conv_vae_model: A variational auto-encoder model implemented as PyTorch Module
    :type conv_vae_model: cnn.convvaemodel.ConvVAEModel
    :param hyper_params: Instance of hyper-parameters for this variational auto-encoder
    :type hyper_params: nnet.hyperparams.HyperParams
    :param debug: Optional debugging method
"""


class ConvVAE(VAE):
    def __init__(self,
                 conv_vae_model: ConvVAEModel,
                 hyper_params: HyperParams,
                 debug):
        super(ConvVAE, self).__init__(conv_vae_model, hyper_params, debug)

    def loss_func(
            self,
            predicted: torch.Tensor,
            actual: torch.Tensor,
            mu: torch.Tensor,
            log_var: torch.Tensor) -> float:
        """
            Define the loss function for convolutional variational auto encoder
            :param predicted: Predicted input_tensor
            :param actual: label input_tensor
            :param mu: mean from the normal distribution on z-space
            :param log_var: Log of variance from the normal distribution on z-space
            :return: Aggregated loss (Reconstruction + KL divergence)
        """
        criterion = self.hyper_params.loss_function
        sz = self.vae_model.input_channels
        try:
            # Cross-entropy for reconstruction loss for binary values
            # and MSE for continuous (TF-IDF) variable
            constants.log_size(actual, 'actual before loss')
            constants.log_size(predicted, 'predict_x')

            # Flatten the input_tensors representing actual values and predicted values
            x_value = actual.view(-1, sz).squeeze(1)
            x_predict = predicted.view(-1, sz).squeeze(1)

            constants.log_size(x_value, 'x_value')
            constants.log_size(x_predict, 'x_predict')
            reconstruction_loss = criterion(x_predict, x_value)

            # Compute the aggregate loss
            return VAE.compute_loss(reconstruction_loss, mu, log_var, sz)
        except RuntimeError as e:
            constants.log_error(f'Runtime error {str(e)}')
            return -1.0
        except ValueError as e:
            constants.log_error(f'Value error {str(e)}')
            return -1.0
        except KeyError as e:
            constants.log_error(f'Key error {str(e)}')
            return -1.0



