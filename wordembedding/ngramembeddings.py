__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import time
import numpy
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Subset
from wordembedding.ngramembeddingsparams import NGramEmbeddingsParams
from wordembedding.ngramcbowmodel import NGramCBOWModel
from util.plottermixin import PlotterParameters
from util.plottermixin import PlotterMixin
from wordembedding.context import Context
from util.s3util import S3Util
from util.logger import Logger
from models.encoder.vocabulary import Vocabulary
import constants
from models.nnet.snapshot import Snapshot
from models.nnet.earlystoplogger import EarlyStopLogger
from datasets.ngramdataset import NGramDataset

"""
    Embedding of NGrams with variable stride. 
    Symmetrical context   word(i-stride), ... word(i-1), target_word, word(i+1), ... word(i+stride)
    Asymmetrical context  word(i-stride), ... word(i-1), target_word
    
    :param  List of valid words used as input_tensor to the wordembedding
    :param ngrams_embeddings_params Configuration parameters for the wordembedding
    :param model_id Identifier for the encoder_model
    :param load_snapshot Boolean flag to specify if the class should be instantiated from a snapshot
"""


class NGramEmbeddings(PlotterMixin):
    def __init__(self,
                 vocabulary: Vocabulary,
                 ngrams_embeddings_params: NGramEmbeddingsParams,
                 model_id: str,
                 load_snapshot: bool = False):
        self.ngrams_embeddings_params = ngrams_embeddings_params
        self.model_id = Logger.s3path_model_id(model_id)
        self.vocabulary = vocabulary
        self.load_snapshot = load_snapshot

    '''
        Generic method for training and evaluation 
        :param ngram_dataset Features,label data set for NGram wordembedding
    '''
    def train_and_eval(self, ngram_dataset: NGramDataset):
        train_set, val_set = NGramEmbeddings.__split_data_loader(ngram_dataset, False)
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.__execute(train_set, val_set)

    def train_from_text(self, text: str):
        words = text.split()
        self.__execute(words)

    @staticmethod
    def infer(input: str, vocab: Vocabulary, embeddings: nn.Embedding, ngram_stride: int) -> list:
        context = Context(ngram_stride, True)
        ngrams = context.apply(input)

        predicted = []
        with torch.no_grad():
            for context, _ in ngrams:
                indexed_context = Variable(torch.tensor(
                    [vocab.word_index_dict[w] for w in context],
                    dtype=torch.long)
                )
                vec = embeddings(indexed_context)
                predicted.append(vec)
        return predicted

    @staticmethod
    def load(model_id: str) -> (nn.Embedding, int):
        model_path = f'{constants.models_path}/{model_id}'
        # Step 1: Load the configuration attributes for the wordembedding
        with open(f'{model_path}.cfg', "r") as config_file:
            content = config_file.read()
            attributes = content.split(",")
            model = NGramCBOWModel.load(int(attributes[0]), int(attributes[1]))
            ngram_stride_len = int(attributes[2])

        # Step 2: Load the weights for the wordembedding
        parameters = torch.load(model_path)
        model.load_state_dict(parameters)
        return model, ngram_stride_len


        # ------------------------------
        #  Private, helper methods
        # ------------------------------


    def __execute(self, train_set: NGramDataset, val_set: NGramDataset):
        vocab_size = len(self.vocabulary.word_index_dict)

        model = NGramCBOWModel(vocab_size, self.ngrams_embeddings_params)
        optimizer = optim.Adam(model.parameters(), self.ngrams_embeddings_params.learning_rate)
        early_stopping = EarlyStopLogger(self.ngrams_embeddings_params.early_stop_patience)

        # If the mode is to be loaded from an existing snapshot....
        snapshot = None
        if self.load_snapshot:
            snapshot = Snapshot(S3Util(constants.default_s3_bucket_name, self.__snapshot_folder()))
            model, optimizer = snapshot.get()

        # Applying the negative Log likelihood loss to the Log-Softmax output layer simulate the cross entropy loss
        loss_function = nn.NLLLoss()
        start = time.time()
        epoch = 0
        early_stopped = False

        # Number of allowed epochs not
        while epoch < self.ngrams_embeddings_params.epochs and not early_stopped:
            train_loss = NGramEmbeddings.__train(train_set, model, optimizer, loss_function)
            train_duration = time.time() - start
            start = time.time()
            accuracy, eval_loss = self.__eval(val_set, model, loss_function)
            eval_duration = time.time() - start
            msg = f'Epoch: {epoch+1} Duration training {train_duration} secs.' \
                  + f' Duration evaluation {eval_duration} secs.' \
                  + f' Est. duration {self.ngrams_embeddings_params.epochs*(train_duration + eval_duration)/60.0} min.'
            constants.log_info(msg)

            # Update the counters and evaluate early stop condition
            early_stopped = early_stopping.evaluate(epoch, train_loss, eval_loss, accuracy)
            # Record a new snapshot if no early stopped
            if not early_stopped:
                self.__snapshot(model, optimizer, snapshot)
            epoch += 1

        # Upon completion of the epochs.. save the encoder_model locally
        self.__save(model, optimizer)
        early_stopping.summary(self.__plotting_params())
        del model, optimizer, early_stopping

    @staticmethod
    def __train(train_set: NGramDataset,
                model: nn.Module,
                optimizer: optim.Optimizer,
                loss_function: nn.Module) -> float:
        total_loss = 0
        for context_t, target_t in train_set:
            for param in model.parameters():
                param.grad = None

            log_probabilities = model(Variable(context_t))  # Implicit invocation to forward
            loss = loss_function(log_probabilities, Variable(target_t.unsqueeze(0)))
                # Compute gradient
            loss.backward(retain_graph=True)
                # encoder+1 = encoder - grad.eps
            optimizer.step()
                # aggregate loss
            total_loss += loss.item()
        return total_loss / len(train_set)


    def __eval(self,  valid_set: NGramDataset, model: nn.Module, loss_function: nn.Module) -> (float, float):
        total_accurate_count = 0.0
        valid_loss = 0.0

        with torch.no_grad():
            for context_t, target_t in valid_set:
                for param in model.parameters():
                    param.grad = None
                log_probabilities = model(Variable(context_t))  # Implicit invocation to forward
                loss = loss_function(log_probabilities, Variable(target_t.unsqueeze(0)))
                valid_loss += loss.item()
                _, predicted = torch.max(log_probabilities.data, 1)
                total_accurate_count += self.__get_accuracy(log_probabilities, target_t)

        stats = total_accurate_count/len(valid_set), valid_loss/len(valid_set)
        return stats


    @staticmethod
    def __get_accuracy(log_probs: torch.Tensor, target_t: torch.Tensor) -> float:
        _, predicted = torch.max(log_probs.data, 1)
        pi = predicted.item()
        ti = target_t.item()
        if pi == ti:
            return 1.0
        else:
            # predicted_word = self.vocabulary.index_word_dict[pi]
            # target_word = self.vocabulary.index_word_dict[ti]
            return 0.0

    @staticmethod
    def __split_data_loader(ngram_dataset: NGramDataset, is_split: bool) -> (NGramDataset, NGramDataset):
        num_records = len(ngram_dataset)
        if is_split:
            split_type = 'random'
            num_train_records = int(num_records * constants.train_eval_ratio)
            num_val_records = num_records - num_train_records
            train_set, val_set = torch.utils.data.random_split(ngram_dataset, [num_train_records, num_val_records])
        else:
            split_type = 'sliced'
            num_train_records = num_records-1
            num_val_records = int(num_records * (1.0 -constants.train_eval_ratio))
            train_set = Subset(ngram_dataset, numpy.arange(num_train_records))
            val_set = Subset(ngram_dataset, numpy.arange(num_val_records))
        constants.log_info(f'Loaded {num_train_records} training and {num_val_records} validation records with {split_type} split')
        return train_set, val_set

    def __save(self, model: NGramCBOWModel, optimizer: optim.Adam):
        _model_path = f'{constants.models_path}/encoder_model-{self.model_id}'
        constants.log_info(f'Save to {_model_path}')

        with open(f'{_model_path}.cfg', "w") as config_file:
            content = f'{len(self.vocabulary.word_index_dict)},{self.ngrams_embeddings_params.embedding_size},{self.ngrams_embeddings_params.ngram_stride}'
            config_file.write(content)
        torch.save(model.embeddings.state_dict(), _model_path)
        torch.save(optimizer.state_dict(), _model_path + '_optimizer')

    def __snapshot(self, model: NGramCBOWModel, optimizer: optim.Adam, snapshot: Snapshot):
        if self.ngrams_embeddings_params.is_record_snapshot and snapshot is not None:
            snapshot.put(model.state_dict, 'mod')
            snapshot.put(optimizer.state_dict, 'opt')

    def __snapshot_folder(self):
        return f'temp/snapshot-{self.model_id}'

    def __plotting_params(self) -> list:
        return [
            PlotterParameters(self.ngrams_embeddings_params.epochs, '', 'training loss',
                              '2 layer CBOW embedding 256x48'),
            PlotterParameters(self.ngrams_embeddings_params.epochs, 'epoch', 'eval loss', ''),
            PlotterParameters(self.ngrams_embeddings_params.epochs, 'epoch', 'metric', '')
        ]

