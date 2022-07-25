__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import torch.nn as nn
import constants
from torch.nn import functional as F
from wordembedding.embeddedngrams import EmbeddedNGrams
from wordembedding.ngramembeddingsparams import NGramEmbeddingsParams

"""
    Implementation of plain vanilla NGram word2vec contextual bag of words.
    This class reuses feature_name elements of the base class EmbeddedNGrams. Unfortunately, 
    
    :param vocab_size Size of the vocab
    :param ngrams_embeddings_params Parameters for the wordembedding
"""


class NGramCBOWModel(EmbeddedNGrams):
    def __init__(self, vocab_size: int, ngrams_embeddings_params: NGramEmbeddingsParams):
        super(NGramCBOWModel, self).__init__(
            vocab_size,
            ngrams_embeddings_params.embedding_size,
            ngrams_embeddings_params.ngram_stride-1,
            ngrams_embeddings_params.is_symmetric
        )
        # Decoder modules
        self.linear1 = nn.Linear(self.dim, ngrams_embeddings_params.input_connected_layer_size)
        self.linear2 = nn.Linear(ngrams_embeddings_params.input_connected_layer_size, vocab_size)

    @staticmethod
    def load(vocab_size: int, embedding_size: int) -> nn.Embedding:
        return nn.Embedding(vocab_size, embedding_size)

    def forward(self, input: torch.Tensor) -> float:
        # Just to force to use CUDA if available
        embeds = self.embed(input)
        # This can be replaced by a convolutional layer
        out1 = self.linear1(embeds)
        ou2 = F.relu(out1)
        out = self.linear2(ou2)
        # Flatten to match the size of the input_tensor input_tensor, then apply a soft max
        log_prob =  F.log_softmax(out, dim=1)
        return log_prob


    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        self.embeddings.weight = nn.Parameter(self.embeddings.weight.to(constants.torch_device))
        x = self.embeddings(inputs)
        return x.view((1, -1))

    def __repr__(self) -> str:
        return '\n'.join([str(self.embeddings), str(self.linear1), str(self.linear2)])
