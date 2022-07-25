__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
from wordembedding.ngramembeddingsparams import NGramEmbeddingsParams
from wordembedding.ngramembeddings import NGramEmbeddings
from models.encoder.vocabulary import Vocabulary
from models.encoder.loader import Loader
import constants
from wordembedding.context import Context
from datasets.ngramdataset import NGramDataset


class TrainingExec(object):
    @staticmethod
    def vae(arguments: list):
   # s3_folder_name = 'nlp/test1/7/6'
        s3_folder = arguments[1]
        loader = Loader(s3_folder, constants.default_s3_bucket_name)
        code_sources_dataset = EmbeddedVecDataset(loader.df)
        sz = code_sources_dataset[2].size()

        input_layer_size = sz[0] * sz[1]
        enc_layer_sizes = [input_layer_size, 512, 128, 32]
        activations = [torch.nn.LeakyReLU(), torch.nn.LeakyReLU()]
        encoder = Encoder(enc_layer_sizes, activations, False)

        dec_layer_sizes = [32, 128, 512, input_layer_size]
        activations = [torch.nn.LeakyReLU(), torch.nn.LeakyReLU(), torch.nn.Sigmoid()]
        decoder = Decoder(dec_layer_sizes, activations)

        epochs = 50
        learning_rate = 0.0002
        vae_code_model = LinearVAE(encoder, decoder, code_sources_dataset, True)
        vae_code_model.train_and_eval(epochs, learning_rate)

    @staticmethod
    def create_embeddings(arguments: list) -> list:
        data_source = arguments[1]
        config_label = arguments[2]
        limit = int(arguments[3])
        is_snapshot = bool(arguments[4])

        terms_key = 'doc_terms_str'
        ngrams_embeddings_params = TrainingExec.config(config_label, is_snapshot)
        constants.log_info(f'Embedding params:\n{repr(ngrams_embeddings_params)}')
        ngram_embeddings = NGramEmbeddings(Vocabulary(), ngrams_embeddings_params, data_source)
        s3_path = f'{constants.embedding_layer_s3_folder}/{data_source}/{terms_key}'
        constants.log_info(f'S3 Path: {s3_path}')
        ngram_dataset = NGramDataset.init(constants.default_s3_bucket_name, s3_path, 'doc_terms_str', Context(4), limit)
        constants.log_info('NGram Dataset initialized')
        return ngram_embeddings.train_and_eval(ngram_dataset)

    @staticmethod
    def config(label: str, is_snapshot: bool) -> NGramEmbeddingsParams:
        if label == '1':
            learning_rate = 0.0003
            epochs = 20
            ngram_stride = 4
            embedding_size = 64
            hidden_layer_size = 256
            early_stop_ratio = 1.4
            batch_size = 32
            is_symmetric = True
            is_snapshot_enabled = False
        elif label == '2':
            learning_rate = 0.0003
            epochs = 10
            ngram_stride = 4
            embedding_size = 32
            hidden_layer_size = 128
            early_stop_ratio = 1.4
            batch_size = 32
            is_symmetric = True
            is_snapshot_enabled = False
        elif label == '3':
            learning_rate = 0.001
            epochs = 10
            ngram_stride = 5
            embedding_size = 32
            hidden_layer_size = 128
            early_stop_ratio = 1.4
            batch_size = 32
            is_symmetric = True
            is_snapshot_enabled = False
        elif label == 'test1':
            learning_rate = 0.001
            epochs = 22
            ngram_stride = 4
            embedding_size = 48
            hidden_layer_size = 128
            early_stop_ratio = 12.0
            batch_size = 16
            is_symmetric = True
            is_snapshot_enabled = False
        else:
            learning_rate = 0.0005
            epochs = 15
            ngram_stride = 4
            embedding_size = 32
            hidden_layer_size = 128
            early_stop_ratio = 1.4
            batch_size = 32
            is_symmetric = True
            is_snapshot_enabled = False

        return NGramEmbeddingsParams(
            learning_rate,
            epochs,
            ngram_stride,
            embedding_size,
            hidden_layer_size,
            early_stop_ratio,
            batch_size,
            is_symmetric,
            is_snapshot_enabled)