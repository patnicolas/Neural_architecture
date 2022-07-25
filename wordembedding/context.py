__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import pandas as pd


def symmetric(bag_of_words: list, last_index: int) -> list:
    num_words = len(bag_of_words)
    return [(bag_of_words[i - last_index:i] + bag_of_words[i + 1:i + last_index + 1], bag_of_words[i]) for i in range(last_index, num_words - last_index)]


def asymmetric(words_set: list, last_index: int) -> list:
    lst = []
    for idx in range(len(words_set)):
        bag_of_words = words_set[idx]
        num_words = len(bag_of_words)
        lst.append([(bag_of_words[i: i + last_index], bag_of_words[i + last_index]) for i in range(num_words - last_index)])
    return lst


def split(s) -> list:
    return str(s).split('|')

"""
    Define the context of a word as either Symmetric or Asymmetric around a target word
    :param ngram_stride Stride of N-Gram. Size of the context or number of words prior to the target word
    :param is_symmetric Flag to specify the context is symmetric or asymmetric
"""


class Context(object):
    def __init__(self, ngram_stride: int, is_symmetric: bool = True):
        self.ngram_stride = ngram_stride
        self.is_symmetric = is_symmetric


    def apply_df(self, text_df: pd.DataFrame) -> pd.DataFrame:
        """
            Apply the extraction of context to this data frame
            :param text_df Data frame of text (string) with '|' delimiters
            :return: Data frame of CBOW
        """
        df = text_df.apply(func=lambda s:split(s), axis=1)
        last_index = self.ngram_stride - 1
        if self.is_symmetric:
            return df.apply(lambda s: symmetric(s, last_index))
        else:
            return df.apply(lambda s: asymmetric(s, last_index))

    def apply_lst(self, text_lst: list) -> list:
        xs = [split(text) for text in text_lst]
        last_index = self.ngram_stride - 1
        if self.is_symmetric:
            return [symmetric(ngrams, last_index) for ngrams in xs]
        else:
            return [asymmetric(ngrams, last_index) for ngrams in xs]


    def apply(self, text: str) -> list:
        """
             Apply the extraction of context a single string
             :param text input_tensor string containing '|' delimiters
        """
        s = text.split('|')
        last_index = self.ngram_stride - 1
        if self.is_symmetric:
            return symmetric(s, last_index)
        else:
            return asymmetric(s, last_index)
