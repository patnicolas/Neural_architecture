__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import constants
from torch.utils.data import Dataset, DataLoader
from models.nnet.hyperparams import HyperParams
from models.nnet.earlystoplogger import EarlyStopLogger
from util.plottermixin import PlotterMixin, PlotterParameters
from abc import abstractmethod


"""
    Generic Neural Network abstract class. There are 2 version of train and evaluation
    - _train_and_evaluate Training and evaluation from a pre-configure train loader
    -  train_and_evaluate Training and evaluation from a raw data set
    The method transform_label has to be overwritten in the inheriting classes to support
    transformation/conversion of labels if needed.
    The following methods have to be overwritten in derived classes
    - transform_label Transform the label input_tensor if necessary
    - model_label Model identification
    
    :param hyper_params: Training parameters
    :type hyper_params: nnet.hyperparams.HyperParams
"""


class NeuralNet(PlotterMixin):
    def __init__(self, hyper_params: HyperParams, debug):
        self.hyper_params = hyper_params
        self._debug = debug


    @abstractmethod
    def train_and_eval(self, dataset: Dataset):
        pass

    @abstractmethod
    def train_then_eval(self, train_loader: DataLoader, test_loader: DataLoader):
        pass

    @abstractmethod
    def apply_debug(self, features: list, labels: list, title: str):
        """
            Override the abstract method defined in NeuralNet class and return the original labels
            :param features: List of encoder input_tensors
            :param labels: Labels
            :param title: Title for the debugging info
        """
        pass

    @abstractmethod
    def model_label(self) -> str:
        pass

    def _train_then_eval(self, train_loader: DataLoader, test_loader: DataLoader, model: torch.nn.Module):
        """
            Train and evaluation of a neural network given a data loader for a training set, a
            data loader for the evaluation/test1 set and a encoder_model. The weights of the various linear modules
            (neural_blocks) will be initialize if self.hyper_params using a Normal distribution
            :param train_loader: Data loader for the training set
            :param test_loader:  Data loader for the valuation set
            :param model: Torch encoder_model such as ConvModel, LstmModel, DffModel...
        """
        torch.manual_seed(42)
        self.hyper_params.initialize_weight(list(model.modules()))

        # Create a train loader from this data set
        optimizer = self.hyper_params.optimizer(model)
        early_stop_logger = EarlyStopLogger(self.hyper_params.early_stop_patience)

        # Train and evaluation process
        for epoch in range(self.hyper_params.epochs):
            # Set training mode and execute training
            train_loss = self.__train(optimizer, epoch, train_loader, model)
            # constants.log_info(f'Epoch # {epoch} training loss {train_loss}')
            # Set evaluation mode and execute evaluation
            eval_loss, ave_accuracy = self.__eval(epoch, test_loader, model)
            # constants.log_info(f'Epoch # {epoch} eval loss {eval_loss}')
            early_stop_logger(epoch, train_loss, eval_loss, ave_accuracy)
        # Generate summary
        early_stop_logger.summary(self.__plotting_params())
        del early_stop_logger


    def _train_and_eval(self, dataset: Dataset, model: torch.nn.Module):
        """
            Train and evaluation of a neural network given a data set. This methods invoke _train_and_eval methods
            with train and validation data loser
            :param dataset: Data set containing both training and evaluation set
            :param model: Model such as ConvModel...
        """
        # Create a train loader from this data set
        train_loader, test_loader = NeuralNet.init_data_loader(self.hyper_params.batch_size ,dataset)
        NeuralNet._train_then_eval(self, train_loader, test_loader, model)


    @staticmethod
    def forward(features: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        with torch.no_grad():
            try:
                return model(features)
            except RuntimeError as e:
                constants.log_error(e)
            except AttributeError as e:
                constants.log_error(e)
            except Exception as e:
                constants.log_error(e)


    @staticmethod
    def init_data_loader(batch_size: int, dataset: Dataset) -> (DataLoader, DataLoader):
        torch.manual_seed(42)

        _len = len(dataset)
        train_len = int(_len * constants.train_eval_ratio)
        test_len = _len - train_len
        train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])
        constants.log_info('Extract training and test1 set')

        # Finally initialize the training and test1 loader
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=batch_size,
                                 shuffle=True)
        return train_loader, test_loader


    def compute_accuracy(self, predictionBatch: torch.Tensor, labelsBatch: torch.Tensor, is_last_epoch: bool) -> float:
        return -1.0


    #  ---------------------   Private methods -------------------------


    def __train(self,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                train_loader: DataLoader,
                model: torch.nn.Module) -> float:
        total_loss = 0
        model.train()
        # Initialize the gradient for the optimizer
        loss_function = self.hyper_params.loss_function

        for idx, (features, labels) in enumerate(train_loader):
            try:
                # Reset the gradient to zero
                for params in model.parameters():
                    params.grad = None

                self.apply_debug(features, labels, self.model_label())

                predicted = model(features)  # Call forward - prediction
                raw_loss = loss_function(predicted, labels)
                # Set back propagation
                raw_loss.backward(retain_graph = True)
                total_loss += raw_loss.data
                optimizer.step()
            except RuntimeError as e:
                constants.log_error(str(e))
            except AttributeError as e:
                constants.log_error(str(e))
            except Exception as e:
                constants.log_error(str(e))
        return total_loss / len(train_loader)


    def __eval(self, epoch: int, test_loader: DataLoader, model: torch.nn.Module) -> (float, float):
        total_loss = 0
        total_accuracy = 0
        loss_func = self.hyper_params.loss_function
        model.eval()

        # No need for computing gradient for evaluation (NO back-propagation)
        with torch.no_grad():
            for idx, (features, labels) in enumerate(test_loader):
                try:
                    self.apply_debug(features, labels, self.model_label())
                    predicted = model(features)
                    is_last_epoch = epoch == self.hyper_params.epochs-1
                    accuracy = self.compute_accuracy(predicted, labels, is_last_epoch)
                    if accuracy >= 0.0:
                        total_accuracy += accuracy
                    loss = loss_func(predicted, labels)
                    total_loss += loss.data
                except RuntimeError as e:
                    constants.log_error(e)
                except AttributeError as e:
                    constants.log_error(e)
                except Exception as e:
                    constants.log_error(e)

        average_loss = total_loss / len(test_loader)
        average_accuracy = total_accuracy/len(test_loader)
        return average_loss, average_accuracy

    def __plotting_params(self) -> list:
        return [
            PlotterParameters(self.hyper_params.epochs, '', 'training loss', self.model_label()),
            PlotterParameters(self.hyper_params.epochs, 'epoch', 'eval loss', ''),
            PlotterParameters(self.hyper_params.epochs, 'epoch', 'accuracy', '')
        ]

    def __repr__(self) -> str:
        return repr(self.hyper_params)