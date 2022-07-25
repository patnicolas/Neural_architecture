
from models.encoder.vocabulary import Vocabulary
from wordembedding.ngramembeddings import NGramEmbeddings
from util.logger import Logger


class PredictionExec(object):

    @staticmethod
    def embeddings(arguments: list) -> list:
        input_path = arguments[1]
        model_path = arguments[2]
        vocabulary = Vocabulary()
        embeddings_model, context_width = NGramEmbeddings.load(model_path)
        print(str(embeddings_model))
        logging = Logger()
        input_text = Logger.to_text()
        embedded_list = NGramEmbeddings.infer(input_text, vocabulary, embeddings_model, context_width)
        return [embedded.detach().numpy() for embedded in embedded_list]