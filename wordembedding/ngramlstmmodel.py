__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch.nn as nn
from torch.nn import functional as F
from wordembedding.embeddedngrams import EmbeddedNGrams


class NGramLSTMModel(EmbeddedNGrams):
    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int, is_symmetric: bool):
        super(NGramLSTMModel, self).__init__(vocab_size, embedding_dim, context_size, is_symmetric)
        self.lstm = nn.LSTM(self.dim, 128)
        self.linear = nn.Linear(128, vocab_size)

    def forward(self, inputs: list) -> float:
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.lstm(embeds)
        out = self.linear(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs