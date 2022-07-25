__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import json_normalize
import constants

"""
    Input data format (JSON) format 1
    [
       'doc_terms_str': [
            {'name':'MRNUM', .... },
            {'name':'BILLING', .... },
            ....
        ],
        'doc_terms_str': [
            {'name':'NAME', ... },
            {'name':'OSYK', .... },
        ],
        ....
    ]
    
    Input data format 2 
    [
       ['MRNUM', 'BILLING', ...],
       ['NAME', 'OSYK', ..]
    ]
"""


class TfIdfEncoder(object):
    def __init__(self, input_df: pd.DataFrame, key: str):
        mention_counts_df = input_df[0]
        self.mention_count_tensors = []
        text = []
        for mention_count in mention_counts_df:
            mention_name_count_df = json_normalize(mention_count)
            mention_name_counts = mention_name_count_df[key].values
            entry = ' '.join(mention_name_counts)
            text.append(entry)

        tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(text)
        self.tokens = tfidf_vectorizer.get_feature_names()
        self.count = len(text)
        del text
        self.tfidf_df = pd.DataFrame(
            data=tfidf.toarray(),
            index=np.arange(self.count),
            columns=self.tokens)

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.tfidf_df.values).float().to(constants.torch_device)

    def vocabulary(self) -> list:
        return list(set(self.tokens))


    @staticmethod
    def extract(tfidf_vectorizer: TfidfVectorizer, doc_terms: str, max_num_terms: int = None) -> (bool, list):
        """
            Compute the tf-idf values for terms in the document doc_terms_str. The terms are
            filtered by decreasing tf-idf values and the 'num_terms' top terms are selected
            if num_terms is not None
            :param tfidf_vectorizer: TF-IDF vectorizer
            :param doc_terms: List of document terms
            :param max_num_terms: Maximum number of terms to be extracted if not None
            :return: List of pair (terms_indices, tf-idf weights)
        """
        def sort_func_weight(x: tuple):
            return x[1]

        def sort_func_index(x: tuple):
            return x[0]

        vec = tfidf_vectorizer.transform([doc_terms]).toarray()
        squeezed_vec = vec.squeeze(0)
        # 2- Extract indices of non-zero tfidf values
        non_zero_indices = np.nonzero(squeezed_vec)[0]

        # 3- Select the top num_features the tf-idf weights
        tfidf_weights = []
        [tfidf_weights.append((idx, squeezed_vec[idx])) for idx in non_zero_indices]
        # Need to rank and filter only if the number of terms exceeds
        if max_num_terms is not None and len(tfidf_weights) < max_num_terms:
            tfidf_weights.sort(key=sort_func_weight, reverse=True)
            top_tfidf_weights = tfidf_weights[0: max_num_terms]
            top_tfidf_weights.sort(key=sort_func_index, reverse=False)
            return True, top_tfidf_weights
        else:
            return False, tfidf_weights
