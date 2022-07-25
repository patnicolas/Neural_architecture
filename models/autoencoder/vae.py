__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from torch.utils.data import Dataset, DataLoader
from models.nnet.hyperparams import HyperParams
from torch.optim import Adam
from util.plottermixin import PlotterParameters
from abc import abstractmethod
from util.plottermixin import PlotterMixin
from tqdm import tqdm
import constants

"""
    Base generic class for the hierarchy of Variational Auto-encoders. Specialized auto-encoder sucha
    as feed-forward and convolutional variational auto-encoder have to specify 
    Design patterns: Bridge
    
    :param vae_model: A variational auto-encoder model implemented as PyTorch Module
    :type vae_model: torch.nn.Module
    :param hyper_params: Instance of hyper-parameters for this variational auto-encoder
    :type hyper_params: nnet.hyperparams.HyperParams
    :param debug: Optional debugging method
"""


class VAE(object):
    def __init__(self, vae_model: torch.nn.Module, hyper_params: HyperParams, debug):
        self.vae_model = vae_model
        self.hyper_params = hyper_params
        self.debug = debug
        self.is_training = True


    def train_and_eval(self, input_dataset: Dataset):

        """
            Train and evaluation method using a data set as input_tensor
            :param input_dataset: Input dataset that contains encoder and labels
            :type input_dataset: torch.utils.data.Dataset
        """
        train_loader = DataLoader(dataset=input_dataset, batch_size=self.hyper_params.batch_size, shuffle=True)
        torch.manual_seed(42)
        encoder_params, decoder_params = self.enc_dec_params()
        encoder_optimizer = Adam(encoder_params,
                                 lr=self.hyper_params.learning_rate,
                                 betas=(self.hyper_params.momentum, 0.999))
        decoder_optimizer = Adam(decoder_params,
                                 lr=self.hyper_params.learning_rate,
                                 betas=(self.hyper_params.momentum, 0.999))

        average_training_loss_history = []

        for epoch in range(self.hyper_params.epochs):
            training_loss = self.__train(epoch, encoder_optimizer, decoder_optimizer, train_loader)
            average_training_loss_history.append(training_loss)

        plotter_parameters = PlotterParameters(self.hyper_params.epochs, 'epoch', 'training loss', 'LinearVAE')
        PlotterMixin.single_plot(average_training_loss_history, plotter_parameters)
        del average_training_loss_history

       #  self._train_and_eval(DataLoader(dataset=input_dataset, batch_size=self.hyper_params.batch_size, shuffle=True))


    def enc_dec_params(self) -> (dict, dict):
        """
            Extract the model parameters for the encoder and decoder
            :returns: pair of encoder and decoder parameters (dictionaries)
        """
        return self.vae_model.encoder_model.parameters(), self.vae_model.decoder_model.parameters()


    def __repr__(self):
        return f'Model:\n{repr(self.vae_model)}\nHyper parameters:\n{repr(self.hyper_params)}'


    @abstractmethod
    def loss_func(
            self,
            predicted: torch.Tensor,
            x: torch.Tensor,
            mu: torch.Tensor,
            log_var: torch.Tensor) -> float:
        pass


    def _train_and_eval(self, train_loader: DataLoader, eval_loader: DataLoader):
        """
            Execute training and evaluation for any variational auto-encoder which type is a sub-class of VAE
            :param train_loader: PyTorch loader for the training data
            :param eval_loader: Pytorch loader for the evaluation data
        """
        torch.manual_seed(42)
        encoder_params, decoder_params = self.enc_dec_params()
        encoder_optimizer = Adam(encoder_params,
                                 lr=self.hyper_params.learning_rate,
                                 betas=(self.hyper_params.momentum, 0.999))
        decoder_optimizer = Adam(decoder_params,
                                 lr=self.hyper_params.learning_rate,
                                 betas=(self.hyper_params.momentum, 0.999))

        average_training_loss_history = []
        average_eval_loss_history = []

        for epoch in range(self.hyper_params.epochs):
            training_loss = self.__train(epoch, encoder_optimizer, decoder_optimizer, train_loader)
            average_training_loss_history.append(training_loss)
            eval_loss = self.__eval(epoch, eval_loader)
            average_eval_loss_history.append(eval_loss)

        plotter_parameters = PlotterParameters(self.hyper_params.epochs, 'epoch', 'training loss', 'LinearVAE')
        self.two_plot(average_training_loss_history, average_eval_loss_history, plotter_parameters)
        del average_training_loss_history, average_eval_loss_history


    @staticmethod
    def reshape_output_variation(shapes: list, z: torch.Tensor) -> torch.Tensor:
        assert 2 < len(shapes) < 5, f'Shape {str(shapes)} for variational auto encoder should have at least 3 dimension'
        return z.view(shapes[0], shapes[1], shapes[2], shapes[3]) if len(shapes) == 4 \
            else z.view(shapes[0], shapes[1], shapes[2])


    @staticmethod
    def compute_loss(reconstruction_loss: float, mu: torch.Tensor, log_var: torch.Tensor, num_records: int) -> float:
        """
            Aggregate the loss of reconstruction and KL divergence between proposed and current Normal distribution
            :param reconstruction_loss: Reconstruction loss for this epoch
            :param mu: Mean of the proposed Normal distribution
            :param log_var: Log of variance of the proposed Gaussian distribution
            :param num_records: Number of records used to compute the reconstruction loss and KL divergence
            :return: Aggregate auto-encoder loss
        """
        kullback_leibler = (-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())) / num_records
        constants.log_info(f"Reconstruction loss {reconstruction_loss} KL divergence {kullback_leibler}")
        return reconstruction_loss + kullback_leibler


    # -------------------------  Private methods ------------------

    def __train(
            self,
            epoch: int,
            encoder_optimizer: torch.optim.Optimizer,
            decoder_optimizer: torch.optim.Optimizer,
            data_loader: DataLoader) -> float:
        self.vae_model.train()
        total_loss = 0

        for features, _ in tqdm(data_loader):
            try:
                for params in self.vae_model.parameters():
                    params.grad = None
                recon_batch, mu, latent_var = self.vae_model(features)
                loss = self.loss_func(recon_batch, features, mu, latent_var)

                loss.backward(retain_graph=True)
                total_loss += loss.data
                encoder_optimizer.step()
                decoder_optimizer.step()
            except RuntimeError as e:
                constants.log_error(str(e))
            except AttributeError as e:
                constants.log_error(str(e))
            except Exception as e:
                constants.log_error(str(e))

        average_loss = total_loss / len(data_loader)
        constants.log_info(f'Training average loss for epoch {str(epoch)} is {average_loss}')
        return average_loss


    def __eval(self, epoch: int, valid_loader: DataLoader) -> float:
        self.vae_model.eval()
        total_loss = 0
        evaluation_size = len(valid_loader)
        with torch.no_grad():
            for features, _ in tqdm(valid_loader):
                recon_batch, mu, log_var = self.vae_model(features)
                loss = self.loss_func(recon_batch, features, mu, log_var)
                total_loss += loss.data
        average_loss = total_loss / evaluation_size
        constants.log_info(f'Evaluation: average loss for epoch {str(epoch)} is {average_loss}')
        return average_loss
