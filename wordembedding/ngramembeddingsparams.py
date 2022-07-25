__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

"""
    Class that wraps the configuration parameters for the NGram wordembedding
    :param learning_rate Learning rate used to generate the embedded vectors
    :param epochs Number of epochs used in training
    :param ngram_stride Stride used in extracting the context words
    :param embedding_size Size of the embedded layer
    :param input_connected_layer_size Size of the hidden layer for the terminating fully connected layer
    :param is_symmetric Boolean flag to specify if the context is symmetric (around the target word)
    :param is_snapshot_enabled Is snapshot enabled
    :param is_symmetric Boolean flag to specify if the context is symmetric (around the target word)
"""


class NGramEmbeddingsParams(object):
    def __init__(self,
                 learning_rate: float,
                 epochs: int,
                 ngram_stride: int,
                 embedding_size: int,
                 input_connected_layer_size: int,
                 early_stop_patience: int,
                 batch_size: int,
                 is_symmetric: bool,
                 is_snapshot_enabled: bool):

        assert 1e-6 <= learning_rate <= 0.01, f'Learning rate {learning_rate} should be [1e-6, 0.01]'
        assert 3 <= epochs <= 50, f'Number of epochs {epochs} should be [3, 50]'
        assert 2 <= ngram_stride <= 10, f'Context stride {ngram_stride} should be [3, 10]'
        assert 8 <= embedding_size <= 256, f'Size of wordembedding {embedding_size} should be [16, 256]'
        assert 1 <= batch_size <= 256, f'Size of batch {batch_size} should be [2, 256]'
        assert 1.05 <= early_stop_patience <= 20.0, f'Early stop ratio {early_stop_patience} should be [1.05, 20.0]'

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.ngram_stride = ngram_stride
        self.embedding_size = embedding_size
        self.input_connected_layer_size = input_connected_layer_size
        self.is_symmetric = is_symmetric
        self.is_record_snapshot = is_snapshot_enabled
        self.early_stop_patience = early_stop_patience
        self.batch_size = batch_size

    def symmetric(self) -> bool:
        return self.is_symmetric

    def __repr__(self) -> str:
        return f'  Learning rate: {self.learning_rate}\n  Number epochs: {self.epochs}\n  Context stride: {self.ngram_stride}' \
        f'\n  Embedding size: {self.embedding_size}\n  Input connected layer size: {self.input_connected_layer_size}' \
        f'\n  Early stop ratio: {self.early_stop_patience}\n  Symmetric context: {self.is_symmetric}\n  Batch size: {self.batch_size}'