from unittest import TestCase

import torch
import constants
from wordembedding.ngramcbowmodel import NGramCBOWModel
from wordembedding.ngramembeddingsparams import NGramEmbeddingsParams


class TestNGramCBOWModel(TestCase):
    def test_forward(self):
        learning_rate = 0.0003
        stride = 2
        epochs = 10
        embedding_size = 8
        hidden_layer_size = 18
        batch_size = 8
        early_stop_ratio = 1.4

        ngrams_embeddings_params = NGramEmbeddingsParams(
            learning_rate,
            epochs,
            stride,
            embedding_size,
            hidden_layer_size,
            early_stop_ratio,
            batch_size,
            is_symmetric=True,
            is_snapshot_enabled=False)

        vocab = [
            "Medical", "emergency", "accident", "broken", "legs", "difficulty", "breathing",
            "and", "bleeding", "blood", "clog", "knee", "injury", "pressure"
        ]
        vocab_size = len(vocab)
        context = ["Medical", "emergency"]
        ngram_CBOW = NGramCBOWModel(vocab_size, ngrams_embeddings_params)
        constants.log_info(repr(ngram_CBOW))
        word_index_dict = {word: i for i, word in enumerate(vocab)}
        context1 = torch.tensor([word_index_dict[w] for w in context], dtype=torch.long,
                                       device=constants.torch_device)
        context2 = torch.tensor([word_index_dict[w] for w in context], dtype=torch.long,
                                device=constants.torch_device)
        context = [context1, context2]

        log_probability = ngram_CBOW.embed(context)
        print(log_probability.detach().numpy())
