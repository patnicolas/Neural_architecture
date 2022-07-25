from unittest import TestCase

import torch

class TestEmbeddedNGrams(TestCase):

    def test_embed(self):
        try:
            vocab_size = 90000
            embedding_dim = 32
            embeddings = torch.nn.EmbeddingBag(vocab_size, embedding_dim, mode='sum', sparse=True)
            print(embeddings.weight.detach().numpy())
            input = torch.tensor([[44737, 12216, 88948,  6269],
                    [ 6142, 48130, 81826, 86382],
                    [ 7498, 63564,  7498, 25517],
                    [53648, 12216, 58033, 35884],
                    [81826, 56885, 88948, 56885]])
            x = embeddings(input)
            print(x.size())
            print(x.detach().numpy())
        except Exception as e:
            self.fail()
