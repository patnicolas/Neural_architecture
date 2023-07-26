__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch.nn

from models.nnet.hyperparams import HyperParams
from datasets.termsctxtfidfdataset import TermsCtxTfIdfDataset
import constants
from autocoding.ffclaimpredictor import FFClaimPredictor
from autocoding.cnnclaimpredictor import CNNClaimPredictor
from autocoding.vaeclaimpredictor import VAEClaimPredictor


def run_vae(data_source: str, comments: str, terms_ctx_tfIdf_dataset: TermsCtxTfIdfDataset):
    hyper_params = HyperParams(
        lr=0.0005,
        momentum=0.85,
        epochs=40,
        optim_label=constants.optim_adam_label,
        batch_size=16,
        early_stop_patience=3,
        loss_function=torch.nn.BCELoss(reduction='sum'),
        normal_weight_initialization=True,
        drop_out=0.0)

    print(f'Start grid Vae for {len(terms_ctx_tfIdf_dataset.training_data)} records')

    input_size = terms_ctx_tfIdf_dataset.dimension
    hidden_dimension = 64
    output_size = 32
    latent_size = 16
    fc_hidden_dim = 24
    flatten_input = output_size
    vae_claim_predictor = VAEClaimPredictor.build(
        input_size,
        hidden_dimension,
        output_size,
        latent_size,
        fc_hidden_dim,
        flatten_input,
        hyper_params)
    vae_claim_predictor.train_and_eval(terms_ctx_tfIdf_dataset)


def run_dff(data_source: str, date: str, terms_ctx_tfIdf_dataset: TermsCtxTfIdfDataset, count: int):
    hyper_params_pivot = HyperParams(
        lr=0.0005,
        momentum=0.85,
        epochs=40,
        optim_label=constants.optim_adam_label,
        batch_size=16,
        early_stop_patience=3,
        loss_function=torch.nn.CrossEntropyLoss(),
        normal_weight_initialization=True,
        drop_out=0.75)

    print(f'Start grid search for {len(terms_ctx_tfIdf_dataset.training_data)} records')
    hidden_layers_list = [[96]]
    learning_rates_list = [0.0007]
    batch_sizes_list = [64]
    FFClaimPredictor.grid_search(
        learning_rates_list,
        batch_sizes_list,
        hidden_layers_list,
        hyper_params_pivot,
        terms_ctx_tfIdf_dataset,
        f'{data_source}-{date}',
        count
    )


def run_cnn(data_source: str, comments: str, terms_ctx_tfIdf_dataset: TermsCtxTfIdfDataset):
    hyper_params_pivot = HyperParams(
        lr=0.0005,
        momentum=0.85,
        epochs=40,
        optim_label=constants.optim_adam_label,
        batch_size=16,
        early_stop_patience=3,
        loss_function=torch.nn.CrossEntropyLoss(),
        normal_weight_initialization=True,
        drop_out=0.9)

    model_id = "conv"
    conv_output_layer_size = 128
    dff_hidden_layer_size = 58

    CNNClaimPredictor.build(
            model_id,
            hyper_params_pivot,
            terms_ctx_tfIdf_dataset,
            conv_output_layer_size,
            dff_hidden_layer_size)


def main():
    execution_mode = "dff"
    data_source = "40/7/utrad-2"
    experiment_number = 3
    date = '01.01.22'
    sample_size = 980000
    terms_ctx_tfIdf_dataset = TermsCtxTfIdfDataset(data_source, sample_size)

    if execution_mode == 'dff':
        run_dff(data_source, date, terms_ctx_tfIdf_dataset, experiment_number)
    elif execution_mode == 'cnn':
        run_cnn(data_source, date, terms_ctx_tfIdf_dataset)
    else:
        run_vae(data_source, date, terms_ctx_tfIdf_dataset)
"""
    data_source = "40/7/utrad"
    comments = f'{constants.optim_adam_label}-{constants.train_eval_ratio}-Hdrs-G10-Tanh-'
    sample_size = 980000
    terms_ctx_tfIdf_dataset = TermsCtxTfIdfDataset(data_source, sample_size)
    hyper_params_pivot = HyperParams(
        lr = 0.0005,
        momentum = 0.85,
        epochs = 60,
        optim_label = constants.optim_adam_label,
        batch_size = 16,
        early_stop_patience = 3,
        loss_function = torch.nn.CrossEntropyLoss(),
        normal_weight_initialization = True,
        drop_out = 0.8)

    print(f'Start grid search for {len(terms_ctx_tfIdf_dataset.training_data)} records')
    hidden_layers_list = [[48]]
    learning_rates_list = [0.0006]
    batch_sizes_list = [32]

    FFClaimPredictor.grid_search(
        learning_rates_list,
        batch_sizes_list,
        hidden_layers_list,
        hyper_params_pivot,
        terms_ctx_tfIdf_dataset,
        f'{data_source}-{comments}'
    )
"""

if __name__ == '__main__':
    main()
