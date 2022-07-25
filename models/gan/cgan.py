__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import constants
from models.gan.dcgenerator import DCGenerator
from models.gan.dcdiscriminator import DCDiscriminator
from models.gan.dcgan import DCGan
from models.nnet.hyperparams import HyperParams
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

"""
    Implementation of the Deep conditional convolutional GAN. 
    This specific GAN relies on labels (Supervised) to compute p(encoder | class)
    Two constructors:
    - Direct instantiation from pre-defined generator and discriminator (__init__)
    - Mirrored instantiation for which the generator is a mirror de-convolutional network to the convolutional
        network used in the discriminator (auto_build)
    Note: The input_tensor for the generator and discriminator are to be extended by the one-hot encoding of the number of
    classes.
    
    :param model_id: Identifier for the GAN
    :param gen: GAN generator
    :param disc: Deep convolutional GAN discriminator
    :param hyper_params: Hyper-parameters used for training and evaluation
    :param num_classes: Number of classes C in the computation of p(x | C)
    :param one_hot_target_func: Function to assemble the one hot encoded of input_tensor + class
    :param shapes: Shapes used
"""


class CGan(DCGan):
    def __init__(self,
                 model_id: str,
                 gen: DCGenerator,
                 disc: DCDiscriminator,
                 hyper_params: HyperParams,
                 num_classes: int,
                 one_hot_target_func,
                 shapes: list,
                 z_dim: int,
                 debug = None):
        assert 0 < num_classes < 64, f'CGAN number of classes {num_classes} is out of bounds [1, 63]'

        super(CGan, self).__init__(model_id, gen, disc, hyper_params, debug)
        self.num_classes = num_classes
        self.one_hot_target_func = one_hot_target_func
        self.shapes = shapes
        self.z_dim = z_dim

    @classmethod
    def transposed_init(cls,
                        model_id: str,
                        conv_dimension: int,
                        conv_neural_blocks: list,
                        output_gen_activation: torch.nn.Module,
                        in_channels: int,
                        out_channels: int,
                        hyper_params: HyperParams,
                        num_classes: int,
                        one_hot_target_func,
                        shapes: list,
                        z_dim: int,
                        debug = None) -> DCGan:
        """
            Generate a Deep convolutional conditional GAN using transposed neural blocks
            :param model_id: Identifier for this deep convolutional GAN
            :param conv_dimension: Dimension of the convolution (1 or 2)
            :param conv_neural_blocks: Neural block used in the convolutional model of discriminator
            :param output_gen_activation: Activation function for the output to the generator
            :param in_channels: Size of input_tensor to the discriminator
            :param hyper_params: Hyper-parameters used for training and evaluation
            :param num_classes: Number of classes used in the computation of p(encoder | class)
            :param one_hot_target_func: Function to extract target of aggregated class and input_tensor
            :param shapes: Shapes of input_tensor (1D or 2D convolution)
            :param z_dim: Dimension of the z-Space
            :return: Conditional deep convolutional GAN
        """
        deep_conv_gan = DCGan.transposed_init(model_id,
                                              conv_dimension,
                                              conv_neural_blocks,
                                              output_gen_activation,
                                              in_channels,
                                              out_channels,
                                              hyper_params,
                                              unsqueeze= True)
        return cls(deep_conv_gan.model_id,
                   deep_conv_gan.gen,
                   deep_conv_gan.disc,
                   hyper_params,
                   num_classes,
                   one_hot_target_func,
                   shapes,
                   z_dim,
                   debug)

    @staticmethod
    def adjust_input_sizes(gen_in_channels: int, disc_in_channels: int, num_classes: int) -> (int, int):
        """
            Compute the number of the discriminator input_tensor channels as in_channels + num_classes and
            input_tensor num_tfidf_features of the generator as z-Space num_tfidf_features + num of classes
            :param gen_in_channels: Dimension in the Z-space
            :param disc_in_channels: Number of input_tensor channels for the discriminator
            :param num_classes: Number of classes
            :return: Pair (adjusted input_tensor num_tfidf_features for generator and in channels for discriminator)
        """
        generator_input_dim = gen_in_channels + num_classes
        discriminator_in_channel = disc_in_channels + num_classes
        return generator_input_dim, discriminator_in_channel


    def train(self, epoch: int, train_loader: DataLoader) -> (float, float, int):
        """
            Training method, for convolutional Discriminator and Generator using Wasserstein distance
            :param epoch: Current epoch (starting at 0)
            :param train_loader: PyTorch data loader for training data
            :return: Tuple (mean loss generator, mean loss discriminator, size of dataset
        """
        mean_gen_loss = 0.0
        mean_disc_loss = 0.0

        for real, labels in tqdm(train_loader):
            # Flatten the batch of targets  from the dataset
            real = real.to(constants.torch_device)
            labels = labels.to(constants.torch_device)
            one_hot_labels = F.one_hot(torch.tensor(labels), self.num_classes)
            target_one_hot_labels = self.one_hot_target_func(one_hot_labels, self.shapes)

            fake, disc_loss = self.__discriminate(real, labels, one_hot_labels, target_one_hot_labels)
            # Keep track of the average discriminator loss
            mean_disc_loss += disc_loss.item()

            gen_loss = self.__generate(fake, target_one_hot_labels)
            mean_gen_loss += gen_loss.item()
        return mean_gen_loss, mean_disc_loss, len(train_loader.dataset)


    def __discriminate(self,
                       real: torch.Tensor,
                       labels: torch.Tensor,
                       one_hot_labels: torch.Tensor,
                       target_one_hot_labels: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        cur_batch_size = len(real)

        # Reset the gradient to zero
        for params in self.disc.parameters():
            params.grad = None
        # Get noise corresponding to the current batch_size
        fake_noise = self.gen.adjusted_noise(cur_batch_size, self.z_dim)

        noise_and_labels = torch.cat((fake_noise, one_hot_labels.type(constants.torch_FloatTensor)), 1)
        fake = self.gen(noise_and_labels)

        fake_target_and_labels = torch.cat((fake.float(), target_one_hot_labels.float()), 1)
        real_target_and_labels = torch.cat(
            (real.type(constants.torch_FloatTensor), target_one_hot_labels.type(constants.torch_FloatTensor)), 1)
        disc_fake_pred = self.disc(fake_target_and_labels)
        disc_real_pred = self.disc(real_target_and_labels)

        disc_fake_loss = self.hyper_params.loss_function(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = self.hyper_params.loss_function(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = 0.5 * (disc_fake_loss + disc_real_loss)
        disc_loss.backward(retain_graph=True)
        self.disc_opt.step()
        return fake, disc_loss


    def __generate(self, fake: torch.Tensor, target_one_hot_labels: torch.Tensor) -> torch.Tensor:
        # Reset the gradient to zero
        for params in self.gen.parameters():
            params.grad = None

        fake_target_and_labels = torch.cat((fake, target_one_hot_labels.float()), 1).float()

        # This will error if you didn't concatenate your labels to your image correctly
        disc_fake_pred = self.disc(fake_target_and_labels)
        gen_loss = self.hyper_params.loss_function(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        self.gen_opt.step()
        # Keep track of the generator losses
        return gen_loss
