from unittest import TestCase

import constants
from wordembedding.ngramembeddings import NGramEmbeddingsParams
from wordembedding.ngramembeddings import NGramEmbeddings
from wordembedding.context import Context
from models.encoder.vocabulary import Vocabulary
from datasets.ngramdataset import NGramDataset


class TestNGramEmbeddings(TestCase):
    def test_train_and_eval(self):
        try:
            print(constants.vocab_path)
            data_source = 'test1/7/3'
            context_stride = 3
            ngram_embeddings = self.__get_ngram_embeddings()
            s3_path = f'{constants.embedding_layer_s3_folder}/{data_source}/doc_terms_str'

            context = Context(context_stride)
            limit = 8
            ngram_dataset = NGramDataset.init(constants.default_s3_bucket_name, s3_path, 'doc_terms_str', context, limit)
            print(f'First element {ngram_dataset[0]}')
            ngram_embeddings.train_and_eval(ngram_dataset)
        except Exception as e:
            self.fail(str(e))

    def test_load(self):
        try:
            model_id = 'test1-7-3'
            embeddings, context_width = NGramEmbeddings.load(model_id)
            assert context_width > 1, f'Context width {context_width} should be > 1'
        except Exception as e:
            self.fail(str(e))

    def test_infer(self):
        try:
            model_id = 'test1-7-3'
            embeddings, context_width = NGramEmbeddings.load(model_id)
            input = "pat|breast|scr|patient|breast|scr|physician|number|breast|scr|for|screen|time|mammogram|for|breast|procedure|bilateral|and|mlo|bilateral|and|mlo|prior|study|bilateral|breast|scr|green|bilateral|breast|scr|green|bilateral|breast|breast|center|tissue|are|scattered|fibroglandular|architectural|microcalcifications|are|skin|nipple|bilateral|scattered|are|similar|prior|postsurgical|scar|the|upper|outer|quadrant|the|left|breast|are|significant|changes|with|prior|markings|skin|open|circle|palpable|line|scar|digital|mammography|and|imaging|and|with|benign|screening|mammogram|cancer|risk|risk|assessment|patient|information|risk|survey|the|time|lifetime|breast|cancer|greater|equal|mammogram|and|screening|breast|mri|high|risk|the|patient|elevated|risk|the|breast|and|ovarian|genetic|counseling|and|with|high|risk|mutation|risk|greater|equal|genetic|and|screening|final|signed|date|and|signed|signed"
            prediction = NGramEmbeddings.infer(input, Vocabulary(), embeddings, context_width)
            print(f'Latent space representation for input:\n{str(prediction)}')
        except Exception as e:
            self.fail(str(e))

    def __get_ngram_embeddings(self):
        data_source = 'test1/7/3'
        context_stride = 3
        batch_size = 5
        num_epochs = 3
        lr = 0.001
        embedding_size = 32
        hidden_layer_size = 128
        early_stop_patience = 4

        ngrams_embeddings_params = NGramEmbeddingsParams(
            lr,
            num_epochs,
            context_stride,
            embedding_size,
            hidden_layer_size,
            early_stop_patience,
            batch_size,
            is_symmetric=True,
            is_snapshot_enabled=False)
        return NGramEmbeddings(Vocabulary(), ngrams_embeddings_params, data_source)
