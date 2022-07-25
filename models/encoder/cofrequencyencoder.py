__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from models.encoder.featureencoder import FeatureEncoder
from models.encoder.vocabulary import Vocabulary
from datasets.s3datasetloader import S3DatasetLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import constants

"""
    Encoder for co-occurence matrix of the top 'max_num_terms' keywords sorted by their tf-idf values.
    The process is as follows
        1. Load data frame containing the keywords (matching a given Vocabulary) previously extracted from a document
        2. Compute the tf-idf values for all the keywords across all the documents
        3. Select the top 'max_num_terms' keywords according to their tf-idf values
        4. Compute the co-occurrence frequency of these keywords with each document
        
    :param text_col_name: Name of the column containing the keywords
    :param vocabulary: Vocabulary of acceptable keywords
    :param max_num_terms: Define the number of keywords with the highest tf-idf values
    :param num_samples: Number of the sample used to encode the keywords
"""


class CoFrequencyEncoder(FeatureEncoder):
    # Static constants
    index_weight_separator = '#'
    index_index_separator = '|'
    first_index_col_name = 'idx1'
    second_indices_col_name = 'assocWeights'

    def __init__(self, text_col_name: str, vocabulary: Vocabulary, max_num_terms: int, num_samples: int = -1):
        super(CoFrequencyEncoder, self).__init__(text_col_name)
        self.vocabulary = vocabulary
        self.max_num_terms = max_num_terms
        self.indices_dict = CoFrequencyEncoder.load_co_frequency_indices(num_samples)

    def name(self) -> str:
        return FeatureEncoder.covariance_encoder_label

    def encode(self, feature_df: pd.DataFrame) -> np.array:
        """
            Encoder for terms co-occurrence normalized frequencies
            :param feature_df: Input data frame
            :return: Numpy array N x N x 1
        """
        text_df = feature_df[self.text_col_name]
        corpus = text_df.values.tolist()
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(corpus)

        co_frequency_np = [self.create_co_frequency_matrix(tfidf_vectorizer, doc_terms, self.indices_dict)
                             for doc_terms in corpus]
        return np.concatenate(co_frequency_np, axis=0)

    def __repr__(self):
        keys = self.indices_dict.keys()
        return f'Max num of terms: {self.max_num_terms}\nFirst few indexed covariance: {list(keys)[0:20]}'

    def create_co_frequency_matrix(self,
                                   tfidf_vectorizer: TfidfVectorizer,
                                   doc_terms: str,
                                   indices_dict: dict) -> np.array:
        """
            Generate the Co-variance matrix from a dictionary of terms indices and covariance weights
                term -> Vocabulary -> term_index
                {term_index1: {term_index2: cov(term1, term2}}
            The steps are
            1 Select the most relevant terms in document (from doc_terms_str) using tf-idf values as ranking
            2 Generate the Covariance matrix using term indices

            :param tfidf_vectorizer: TF-IDF Vectorizer
            :param doc_terms: Concatenate terms selected from a document
            :param indices_dict: Dictionary of indices of terms in vocabulary as loaded from S3
            :return: ND matrix of covariance value between
        """
        from models.encoder.tfidfencoder import TfIdfEncoder
        # Step 1. Select the most relevant terms using tf-idf values as ranking function
        truncated, top_tfidf_weights = TfIdfEncoder.extract(tfidf_vectorizer, doc_terms, self.max_num_terms)
        feature_names = tfidf_vectorizer.get_feature_names()
        relevant_feature_names = [feature_names[idx] for idx, _ in top_tfidf_weights]
        first_index_keys = list(indices_dict.keys())

        # Step 2. Generate the Covariance matrix from the most relevant features
        co_frequency_np = np.zeros((len(feature_names), len(feature_names)), dtype = np.float32)

        def generator(idx1: int, relevant_feature_name1: str, idx2: int, relevant_feature_name2: str):
            if relevant_feature_name1 in self.vocabulary.word_index_dict and \
                relevant_feature_name2 in self.vocabulary.word_index_dict:
                relevant_feature_index1 = self.vocabulary.word_index_dict[relevant_feature_name1]
                relevant_feature_index2 = self.vocabulary.word_index_dict[relevant_feature_name2]
                # Is this relevant index?
                if relevant_feature_index1 in first_index_keys:
                    index2_dict = indices_dict[relevant_feature_index1]
                    if relevant_feature_index2 in list(index2_dict.keys()):
                        weight = index2_dict[relevant_feature_index2]
                        co_frequency_np[idx1][idx2] = weight
            else:
                constants.log_warn(f'{relevant_feature_name1} or {relevant_feature_name2} not in dictionary')

        [generator(idx1, relevant_feature_name1, idx2, relevant_feature_name2)
         for idx1, relevant_feature_name1 in enumerate(relevant_feature_names)
            for idx2, relevant_feature_name2 in enumerate(relevant_feature_names)]
        return co_frequency_np


    @staticmethod
    def load_co_frequency_indices(num_samples: int = -1) -> dict:
        """
            Create a Co-frequency tensor for a given data source. The covariance data defined in the files contained
            in the S3 folder have the following format
              Document 1: {"term_index}, "term_index1#cov(term, term1) | term_index2#cov(term, term2), ... "}
              Document 2: {"term_index}, "term_index1#cov(term, term1) | term_index2#cov(term, term2), ... "}
              ...
            where the term_index is generated from the Vocabulary dictionary {term <-> term_index}

            :param s3_folder: Source of data (S3 folder)
            :param num_samples: Number of records to be processed (all the records if num_samples = -1
            :return: List of tuple (first code index, second code index, weights)
        """

        s3_bucket = constants.s3_config['bucket_name']
        s3_folder = constants.s3_sources['covariance']
        columns = [CoFrequencyEncoder.first_index_col_name, CoFrequencyEncoder.second_indices_col_name]
        constants.log_info(f'Co-occurence columns {str(columns)}')
        s3_data_loader = S3DatasetLoader(s3_bucket, s3_folder, columns, False, '.json', num_samples)
        constants.log_info('Co-occurence data set loaded')

        def extract_gen():
            for idx, assoc_weights in s3_data_loader.df.values.tolist():
                pairs = assoc_weights.split(CoFrequencyEncoder.index_index_separator)
                for pair in pairs:
                    idx2_weights = pair.split(CoFrequencyEncoder.index_weight_separator)
                    yield idx, int(idx2_weights[0]), float(idx2_weights[1])

        index_weight_tuples = extract_gen()
        acc = []
        while True:
            weight_tuple = next(index_weight_tuples, None)
            if weight_tuple is None:
                break
            else:
                acc.append(weight_tuple)
        print(acc)

        idx1_dict = {}

        # Generator for extracting (term_index1, term_index2, weight) tuple
        def generator(idx1: int, idx2: int, weight: float):
            idx2_dict = idx1_dict.get(idx1) if idx1 in idx1_dict.keys() else {}
            idx2_dict.update({idx2: weight})
            idx1_dict.update({idx1: idx2_dict})

        [generator(idx1, idx2, weight) for idx1, idx2, weight in index_weight_tuples]
        return idx1_dict





