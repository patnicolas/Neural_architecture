import torch.nn

from models.nnet.hyperparams import HyperParams
from models.dffn import DFFNeuralBlock
from models.dffn import DFFModel
from models.dffn import DFFNet
from datasets.termsctxtfidfdataset import TermsCtxTfIdfDataset
from autocoding.tuning import Tuning
import constants



class FFClaimPredictor(DFFNet, Tuning):
    def __init__(self, dff_model: DFFModel, hyper_params: HyperParams):
        super(FFClaimPredictor, self).__init__(dff_model, hyper_params, None)

    @classmethod
    def build(
            cls,
            model_id: str,
            hyper_params: HyperParams,
            dataset: TermsCtxTfIdfDataset,
            hidden_layers_size: list) -> DFFNet:

        layers_sizes = [dataset.dimension, hidden_layers_size[0], hidden_layers_size[1], dataset.num_labels] \
            if len(hidden_layers_size) > 1 \
            else \
            [dataset.dimension, hidden_layers_size[0], dataset.num_labels]

        num_layers = len(layers_sizes)
        assert num_layers > 2, f'Number of layers in this network {num_layers} should be > 2'

        # The activation of the output layer is none (pure linear)
        neural_blocks = [
            DFFNeuralBlock(layers_sizes[index], layers_sizes[index + 1], None, 0.0) if index == num_layers - 2
            else DFFNeuralBlock(layers_sizes[index], layers_sizes[index + 1], torch.nn.Tanh(), 0.0)
            for index in range(num_layers - 1)]
        dff_model =  DFFModel(model_id, neural_blocks)
        return cls(dff_model, hyper_params)


    @classmethod
    def update(cls, dff_model: DFFModel, hyper_params: HyperParams):
        return cls(dff_model, hyper_params)


    def train(self, input_data_set: TermsCtxTfIdfDataset, config: str):
        import time
        constants.log_info(f'Configuration: {config} --->\n{str(self.dff_model)}\n{str(self.hyper_params)}')
        start = time.time()
        self.train_and_eval(input_data_set)
        duration = time.time() - start
        constants.log_info(f'Duration: {duration}')


    @classmethod
    def grid_search_params(
            cls,
            lrs: list,
            batch_sizes: list,
            hidden_layers_size: list,
            hyper_params_pivot: HyperParams,
            dataset: TermsCtxTfIdfDataset,
            model_descriptor: str,
            counter: int) -> iter:
        """
            Generic grid search for
            :param model_id: Model identifier
            :param lrs: List of learning rate
            :param batch_sizes:  List of batch sizes
            :param hyper_params_pivot: Hyper parameters
            :param dataset: Original data set
            :param hidden_layers_sizes: List of hidden layers size
            :return: Iterator for grid search
        """
        hyper_params_list = hyper_params_pivot.grid_search(lrs, batch_sizes)

        # Comprehensive list for generic dff models
        model_desc = FFClaimPredictor.__get_model_id(hidden_layers_size, hyper_params_pivot)
        claim_predictor = FFClaimPredictor.build(
            model_desc,
            hyper_params_pivot,
            dataset,
            hidden_layers_size)
        model = claim_predictor.dff_model

        # Generation expression for
        for hyper_param in hyper_params_list:
            model_desc = f'{FFClaimPredictor.__get_model_id(hidden_layers_size, hyper_param)}-{model_descriptor}'
            model.set_model_id(model_desc)
            yield FFClaimPredictor.update(model, hyper_param)


    @classmethod
    def grid_search(
            cls,
            lrs: list,
            batch_sizes: list,
            hidden_layers_sizes: list,
            hyper_params_pivot: HyperParams,
            dataset: TermsCtxTfIdfDataset,
            model_desc: str,
            counter: int) -> iter:
        """
            Generic grid search for
            :param model_id: Model identifier
            :param lrs: List of learning rate
            :param batch_sizes:  List of batch sizes
            :param hyper_params_pivot: Hyper parameters
            :param dataset: Original data set
            :param hidden_layers_sizes: List of hidden layers size
            :return: Iterator for grid search
        """
        for hidden_layers_size in hidden_layers_sizes:
            search_parameters_iter = FFClaimPredictor.grid_search_params(
                lrs,
                batch_sizes,
                hidden_layers_size,
                hyper_params_pivot,
                dataset,
                model_desc,
                counter)

            running = True
            while running:
                try:
                    claim_predictor = next(search_parameters_iter)
                    counter += 1
                    claim_predictor.train(dataset, f'{model_desc}{str(counter)}')
                except StopIteration as e:
                    constants.log_info("Exit grid search")
                    running = False

          #  super().save(output_name)


    @staticmethod
    def __get_model_id(hidden_layers: list, hyper_params: HyperParams) -> str:
        """
            Extract new model description using Network architecture and hyper parameters
            :param hidden_layers: Hidden layers layout
            :param hyper_params: Hyper parameters
            :return: Model descriptor
        """
        return f'{hyper_params.get_label()}-net.{str(hidden_layers)}'


    def compute_accuracy(self, predictionBatch: torch.Tensor, labelsBatch: torch.Tensor, is_last_epoch: bool) -> float:
        """
            Compute the accuracy of the prediction for this particular model
            :param predictionBatch: Batch predicted features
            :param labelsBatch: Batch labels
            :return: Average accuracy for this batch
        """
        accuracy, prediction_label = Tuning.get_accuracy(predictionBatch, labelsBatch)
        if is_last_epoch:
            with open('output/predict_labels.csv', 'w') as f:
                constants.log_info(f'Write {len(prediction_label)} prediction-labels')
                f.write('\n'.join(prediction_label))
        return accuracy

