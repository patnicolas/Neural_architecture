__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

import constants
from util.ioutil import IOUtil

"""
    Implements a simple dual access to a dictionary of doc_terms_str
    :param terms_path: Path to the file containing doc_terms_str
"""


class Vocabulary(object):
    def __init__(self, terms_path: str = constants.vocab_path):
        terms, self.word_index_dict = Vocabulary.__get_word_indices(terms_path)
        self.index_word_dict = {i: word for i, word in enumerate(terms)}
        del terms

    @staticmethod
    def get_index_weights() -> dict:
        """
            Retrieve the weights that normalize the indices of terms
            :return: Dictionary {term, weight = index = len(vocabulary) }
        """
        # Retrieve the map of terms -> indices
        terms, word_index_dict = Vocabulary.__get_word_indices(constants.vocab_path)
        scale_factor = len(word_index_dict)

        # Apply the scale factor
        term_index_weights = {word: float(i)/scale_factor for word, i in word_index_dict.items()}
        # Clean up
        del word_index_dict, terms
        return term_index_weights


    @staticmethod
    def __get_word_indices(terms_path: str) -> (list, dict):
        io_util = IOUtil(terms_path)
        terms = [w.rstrip().lower() for w in io_util.to_lines()]
        terms.sort
        return terms, {word: i for i, word in enumerate(terms)}

    def __len__(self):
        return len(self.word_index_dict)

    def __getitem__(self, key: str):
        """
            Retrieve the index of associated with a medical doc_terms_str from the vocabulary
            :param key: Term or key
            :return: Index associated with the key or None if the key does not exists
        """
        try:
            return self.word_index_dict[key]
        except IndexError as e:
            constants.log_error(str(e))
            None


    def __repr__(self) -> str:
        return str(self.word_index_dict) + '\n' + str(self.index_word_dict)