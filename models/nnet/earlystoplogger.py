__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from util.plottermixin import PlotterMixin
import constants

"""
    Enforce early stopping for any training/evaluation pair of execution and records loss for profiling and 
    summary
    The early stopping algorithm is implemented as follows:
       Step 1: Record new minimum evaluation loss, min_eval_loss
       Step 2: If min_eval_loss < this_eval_loss Start decreasing patience count
       Step 3: If patience count < 0, apply early stopping
    Patience for the early stop is an hyper-parameter. A metric can be optionally recorded if value is >= 0.0
    
    :param patience: Number of time the eval_loss has been decreasing
    :type ratio: int
    :param min_diff: Minimum difference (min_eval_loss - lastest eval loss)
    :type min_diff: float
    :param early_stopping_enabled: Early stopping is enabled if True, disabled otherwise
"""


class EarlyStopLogger(PlotterMixin):
    def __init__(self, patience: int, min_diff: float = -1e-5, early_stopping_enabled: bool = True):
        self.patience = patience
        self.training_losses = []
        self.eval_losses = []
        self.metric = []
        self.min_loss = -1.0
        self.min_diff = min_diff
        self.early_stopping_enabled = early_stopping_enabled


    def __call__(self, epoch: int, train_loss: float, eval_loss: float, metric: float = -10000.0) -> bool:
        """
            Implement the early stop and logging of training, evaluation loss. An metric < 0.0 is not recorded
            :param epoch:  Current epoch index (starting with 1)
            :param train_loss: Current training loss
            :param eval_loss: Current evaluation loss
            :param metric: Select metric (i.e. accuracy, precision,....), if metric == -1, no metric is recorded
            :return: True if early stopping, False otherwise
        """
        # Step 1. Apply early stopping criteria
        is_early_stopping = self.__evaluate(eval_loss)
        # Step 2: Record training, evaluation losses and metric
        self.__record(epoch, train_loss, eval_loss, metric)
        constants.log_info(f'Is early stopping {is_early_stopping}')
        return is_early_stopping


    def summary(self, plotter_parameters: list):
        """
            Summary with plotting capability
            :param plotter_parameters: List of plotter parameters
        """
        if self.metric:
            self.three_plot(self.training_losses, self.eval_losses, self.metric, plotter_parameters)
        else:
            self.two_plot(self.training_losses, self.eval_losses, plotter_parameters)

    # -----------------  Private methods -----------------------

    def __evaluate(self, eval_loss: float) -> bool:
        if self.early_stopping_enabled:
            False
        else:
            is_early_stopping = False
            if self.min_loss < 0.0:
                self.min_loss = eval_loss
            elif self.min_loss - eval_loss > self.min_diff:
                self.min_loss = eval_loss
            elif self.min_loss - eval_loss <= self.min_diff:
                self.patience =- 1
                if self.patience < 0:
                    constants.log_info('Early stopping')
                    is_early_stopping = True
            return is_early_stopping


    def __record(self, epoch: int, train_loss: float, eval_loss: float, accuracy: float):
        metric_str = f', Accuracy: {accuracy}' if accuracy >= 0.0 else ''
        status_msg = f'Epoch: {epoch}, Train loss: {train_loss}, Eval loss: {eval_loss}{metric_str}'
        constants.log_info(status_msg)

        self.training_losses.append(train_loss)
        self.eval_losses.append(eval_loss)
        # If metric is valid, record it
        if accuracy > -10000.0:
            self.metric.append(accuracy)





