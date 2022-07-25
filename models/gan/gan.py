__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import constants
import time
from models.gan.generator import Generator
from models.gan.discriminator import Discriminator
from models.nnet.hyperparams import HyperParams
from models.nnet.earlystoplogger import EarlyStopLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.plottermixin import PlotterMixin, PlotterParameters


"""
    Generic Generic Adversarial Network. Like any Deep learning algorithm, a Generative Adversarial Network has
    - model identifier
    - model (Generator + Discriminator)
    - Hyper-parameters (including optimizer)
    
    Training steps:
    1. Retrieve the target samples from the train loader by batch
    2. Generate target labels with class = 1 (correct) - real_samples
    3. Generate latent space (batch-size, 2 classes) - latent_samples
    4. Compute the synthetic/generated samples through the generator - synthetic_samples
    5. Generated predicted_samples as fail (class 0) - predicted_labels
    6. Concatenate the input_tensor for samples (target and synthetic) - all_samples
    7. Concatenate the input_tensor for labels (target and predicted) - all_labels
    8. Reset the gradient for the Discriminator
    9. Execute Discriminator forward (prediction)
    10. Compute the loss for the Discriminator
    12. Compute the gradients  (backward)
    13. Update the new values
    14. Initialize the input_tensor (latent samples) for the generator - latent_samples
    15. Reset the gradient for the Generator
    16. Execute Generator forward (prediction)
    17. Compute the loss for the Generator
    18. Compute the gradients  (backward)
    
    :param model_id: Model identifier
    :param gen: Generator (~ decoder)
    :param disc: Discriminator (i.e. Binary classifier)
    :param hyper_params: Hyper parameters
"""


class Gan(PlotterMixin):
    def __init__(self, model_id: str, gen: Generator, disc: Discriminator, hyper_params: HyperParams, debug = None):
        self.model_id = model_id
        self.gen = gen.to(device = constants.torch_device)
        self.disc = disc.to(device = constants.torch_device)
        self.hyper_params = hyper_params
        self.gen_opt = self.hyper_params.optimizer(self.gen)
        self.disc_opt = self.hyper_params.optimizer(self.disc)
        self.debug = debug

    def train_and_eval(self, train_loader: DataLoader, eval_loader: DataLoader):
        """
            Training and evaluation using pre-selected data loader. The weights of the generator and
            discriminator are initialized with a normal distribution
            :param train_loader: Loader for the training data
            :param eval_loader: Loader for the evaluation data
            :return: None
        """
        torch.manual_seed(42)
        self.__initialize_weights()
        # Initialize the early stopper
        early_stop_logger_generator = EarlyStopLogger(self.hyper_params.early_stop_patience, -1e-5, False)
        early_stop_logger_discriminator = EarlyStopLogger(self.hyper_params.early_stop_patience, -1e-5, False)

        for epoch in tqdm(range(self.hyper_params.epochs)):
            train_gen_loss, train_disc_loss, train_len = self.train(epoch, train_loader)
            eval_gen_loss, eval_disc_loss, eval_len = self.eval(epoch, eval_loader)

            # Records various losses for generator and discriminator
            early_stop_logger_generator(epoch, train_gen_loss/train_len, eval_gen_loss/eval_len)
            early_stop_logger_discriminator(epoch, train_disc_loss/train_len, eval_disc_loss/eval_len)
        self.__display(early_stop_logger_generator, early_stop_logger_discriminator)


    def __repr__(self):
        return f'{self.model_id}\n{self.gen}\n{self.disc}\nHyperParameter:\n{self.hyper_params}'


    # ----------------------- Supporting private methods ----------------------------


    def train(self, epoch: int, train_loader: DataLoader) -> (float, float, int):
        """
            Polymorphic training method, for generic Discriminator and Generator that can be
            overridden by sub-classes
            :param epoch: Current epoch (starting at 0)
            :param train_loader: PyTorch data loader for training data
            :return: Tuple (mean loss generator, mean loss discriminator, size of dataset
        """
        mean_generator_loss = 0.0
        mean_discriminator_loss = 0.0

        for real, _ in train_loader:
            # Reset the gradient to zero
            for params in self.disc.parameters():
                params.grad = None
            disc_loss = self.__discriminate(real)
            mean_discriminator_loss += disc_loss.item()

            # Apply back-propagation
            disc_loss.backward(retain_graph=True)
            # Update optimizer
            self.disc_opt.step()
            mean_generator_loss += self.__generate(real)
        Gan.__summary('Training', mean_discriminator_loss, mean_generator_loss, epoch, len(train_loader.dataset))
        return mean_generator_loss, mean_discriminator_loss, len(train_loader.dataset)


    def eval(self, epoch: int, eval_loader: DataLoader) -> (float, float, int):
        """
            Polymorphic evaluation method, for generic Discriminator and Generator that can be
            overridden by sub-classes
            :param epoch: Current epoch (starting at 0)
            :param eval_loader: PyTorch data loader for the validation data
            :return: Tuple (mean loss generator, mean loss discriminator, size of dataset
        """
        mean_generator_loss = 0.0
        mean_discriminator_loss = 0.0

        for real, _ in eval_loader:
            disc_loss = self.__discriminate(real)
            # Compute the loss in the discriminator/classifier
            mean_discriminator_loss += disc_loss.item()

            fake = self.__set_noise(real)
            disc_fake_pred = self.disc(fake)
            #
            gen_loss = self.hyper_params.loss_function(disc_fake_pred, torch.ones_like(disc_fake_pred))
            mean_generator_loss += gen_loss.item()

        if self.debug is not None:
            self.debug(disc_fake_pred, 25)
        Gan.__summary('Evaluation', mean_discriminator_loss, mean_generator_loss, epoch, len(eval_loader.dataset))
        return mean_generator_loss, mean_discriminator_loss, len(eval_loader.dataset)


    def __initialize_weights(self):
        """
            Initialize the weights of generator and discriminator modules with random normalized values
        """
        self.hyper_params.initialize_weight(self.gen.modules())
        self.hyper_params.initialize_weight(self.disc.modules())


    @staticmethod
    def __summary(desc: str, mean_discriminator_loss: float, mean_generator_loss: float, epoch: int, len: int):
        mean_discriminator_loss /= len
        mean_generator_loss /= len
        summary = f'{desc}: Epoch: {epoch} Discriminator loss {mean_discriminator_loss} Generator loss {mean_generator_loss}'
        constants.log_info(summary)


    def __discriminate(self, real: torch.Tensor) -> torch.Tensor:
        # Create a predicted from a noise
        fake = self.__set_noise(real)

        disc_fake_pred = self.disc(fake.detach())
        disc_fake_loss = self.hyper_params.loss_function(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = self.disc(real)
        disc_real_loss = self.hyper_params.loss_function(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = 0.5*(disc_fake_loss + disc_real_loss)
        # Compute the loss in the discriminator/classifier
        return disc_loss

    def __generate(self, real: torch.Tensor) -> float:
        # Step 1: Reset gradient to zero
        for params in self.gen.parameters():
            params.grad = None

        # Step 2: Generate noise
        fake  = self.__set_noise(real)
        disc_fake_pred = self.disc(fake)

        # Step 3: Compute the loss and apply back-propagation
        gen_loss = self.hyper_params.loss_function(disc_fake_pred, torch.ones_like(disc_fake_pred))
        mean_gen_loss = gen_loss.item()
        gen_loss.backward()
        self.gen_opt.step()

        # Keep track of the average generator loss
        return mean_gen_loss

    def __set_noise(self, target: torch.Tensor) -> torch.Tensor:
        """
            Initialize the noise input_tensor in the latent space
            :param target: Target or label input_tensor
            :return: Generated noise input_tensor
        """
        noise = self.gen.noise(len(target))
        x = noise.view(len(noise), self.gen.z_dim, 1, 1)
        return self.gen(x)

    def __plotting_params(self, title: str) -> list:
        return [
            PlotterParameters(self.hyper_params.epochs, '', 'training loss', title),
            PlotterParameters(self.hyper_params.epochs, 'epoch', 'eval loss', '')
        ]

    def __display(self, early_stop_logger_generator: EarlyStopLogger, early_stop_logger_discriminator: EarlyStopLogger):
        early_stop_logger_generator.summary(self.__plotting_params(f'{self.model_id} Generator'))
        time.sleep(2)
        early_stop_logger_discriminator.summary(self.__plotting_params(f'{self.model_id} Discriminator'))
        del early_stop_logger_generator, early_stop_logger_discriminator
