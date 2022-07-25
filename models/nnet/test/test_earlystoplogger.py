from unittest import TestCase
from models.nnet.earlystoplogger import EarlyStopLogger


class TestEarlyStopLogger(TestCase):
    def test__call(self):
        try:
            is_early_stop = TestEarlyStopLogger.__generate_early_stop_logger(4, 1, 0.01, 0.02, 0.01)
            self.assertFalse(is_early_stop)
        except Exception as e:
            self.fail(str(e))

    def test__call_2(self):
        try:
            early_stop_logger = EarlyStopLogger(3)
            eval_loss = 0.1
            is_early_stop = early_stop_logger(1, 0.01, eval_loss, 0.01)
            self.assertFalse(is_early_stop)
            self.assertTrue(len(early_stop_logger.training_losses) == 1)
            eval_loss = 0.05
            is_early_stop = early_stop_logger(2, 0.01, eval_loss, 0.014)
            self.assertFalse(is_early_stop)
            eval_loss = 0.03
            is_early_stop = early_stop_logger(2, 0.01, eval_loss, 0.014)
            self.assertFalse(is_early_stop)
            eval_loss = 0.002
            is_early_stop = early_stop_logger(2, 0.01, eval_loss, 0.014)
            self.assertFalse(is_early_stop)
        except Exception as e:
            self.fail(str(e))

    @staticmethod
    def __generate_early_stop_logger(
            patience: int,
            epoch: int,
            train_loss: float,
            eval_loss: float,
            metric: float) -> bool:
        early_stop_logger = EarlyStopLogger(patience)
        is_early_stop = early_stop_logger(epoch, train_loss, eval_loss, metric)
        return is_early_stop